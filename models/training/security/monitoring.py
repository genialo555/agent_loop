#!/usr/bin/env python3
"""
Security Monitoring and Alerting System

This module implements comprehensive monitoring and detection mechanisms
for suspicious patterns, security violations, and anomalous behavior
in dataset processing and model training pipelines.
"""

import os
import time
import json
import smtplib
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import (
    Dict, List, Optional, Union, Any, Set, Protocol, 
    Literal, TypeVar, Generic, Callable, Deque
)
from enum import Enum
import structlog
from pydantic import BaseModel, Field, validator
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

from .prompt_injection_analysis import SecurityThreat, ThreatCategory, SecurityAnalysisResult
from .advanced_detection import DetectionResult, AttackVector
from .guardrails import SecurityViolationError

logger = structlog.get_logger(__name__)

# Type definitions
AlertSeverity = Literal["info", "warning", "error", "critical"]
MonitoringLevel = Literal["basic", "detailed", "comprehensive"]
T = TypeVar('T')


class AlertType(str, Enum):
    """Types of security alerts."""
    SECURITY_VIOLATION = "security_violation"
    PROMPT_INJECTION = "prompt_injection"
    RESOURCE_ABUSE = "resource_abuse"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    REPEATED_ATTACKS = "repeated_attacks"
    SYSTEM_COMPROMISE = "system_compromise"
    DATA_EXFILTRATION = "data_exfiltration"
    THRESHOLD_BREACH = "threshold_breach"


@dataclass(frozen=True)
class SecurityEvent:
    """Individual security event record."""
    timestamp: datetime
    event_type: AlertType
    severity: AlertSeverity
    source: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    content_hash: Optional[str] = None
    client_info: Optional[Dict[str, str]] = None
    mitigation_applied: bool = False


@dataclass(frozen=True)
class SecurityAlert:
    """Security alert with aggregated information."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    events: List[SecurityEvent]
    first_occurrence: datetime
    last_occurrence: datetime
    occurrence_count: int
    affected_sources: Set[str]
    recommended_actions: List[str] = field(default_factory=list)
    is_resolved: bool = False


class SecurityMetrics:
    """Security metrics tracking and calculation."""
    
    def __init__(self, window_size: int = 3600):  # 1 hour window
        self.window_size = window_size
        self.events: Deque[SecurityEvent] = deque()
        self.metrics = defaultdict(int)
        self.hourly_stats = defaultdict(lambda: defaultdict(int))
        self.lock = threading.Lock()
    
    def add_event(self, event: SecurityEvent):
        """Add security event and update metrics."""
        with self.lock:
            self.events.append(event)
            self._cleanup_old_events()
            self._update_metrics(event)
    
    def _cleanup_old_events(self):
        """Remove events outside the time window."""
        cutoff_time = datetime.now() - timedelta(seconds=self.window_size)
        while self.events and self.events[0].timestamp < cutoff_time:
            self.events.popleft()
    
    def _update_metrics(self, event: SecurityEvent):
        """Update various security metrics."""
        hour_key = event.timestamp.strftime("%Y-%m-%d-%H")
        
        # Basic counters
        self.metrics["total_events"] += 1
        self.metrics[f"{event.event_type.value}_count"] += 1
        self.metrics[f"{event.severity}_severity_count"] += 1
        
        # Hourly statistics
        self.hourly_stats[hour_key]["total"] += 1
        self.hourly_stats[hour_key][event.event_type.value] += 1
        self.hourly_stats[hour_key][event.severity] += 1
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current security metrics."""
        with self.lock:
            self._cleanup_old_events()
            
            current_hour = datetime.now().strftime("%Y-%m-%d-%H")
            recent_events = len(self.events)
            
            # Calculate rates
            time_window_hours = self.window_size / 3600
            event_rate = recent_events / time_window_hours if time_window_hours > 0 else 0
            
            # Severity distribution
            severity_counts = defaultdict(int)
            for event in self.events:
                severity_counts[event.severity] += 1
            
            # Attack type distribution
            attack_counts = defaultdict(int)
            for event in self.events:
                attack_counts[event.event_type.value] += 1
            
            return {
                "window_size_hours": time_window_hours,
                "recent_events": recent_events,
                "events_per_hour": event_rate,
                "severity_distribution": dict(severity_counts),
                "attack_type_distribution": dict(attack_counts),
                "current_hour_stats": dict(self.hourly_stats[current_hour]),
                "top_sources": self._get_top_sources(),
                "metrics_timestamp": datetime.now().isoformat()
            }
    
    def _get_top_sources(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top event sources by frequency."""
        source_counts = defaultdict(int)
        for event in self.events:
            source_counts[event.source] += 1
        
        sorted_sources = sorted(
            source_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:limit]
        
        return [{"source": source, "count": count} for source, count in sorted_sources]


class AnomalyDetector:
    """Detects anomalous patterns in security events."""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.learning_window = 24 * 3600  # 24 hours
        self.anomaly_threshold = 2.0  # Standard deviations
        self.logger = structlog.get_logger(__name__)
    
    def update_baseline(self, metrics: Dict[str, Any]):
        """Update baseline metrics for anomaly detection."""
        timestamp = datetime.now()
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                if metric_name not in self.baseline_metrics:
                    self.baseline_metrics[metric_name] = {
                        "values": deque(maxlen=100),  # Keep last 100 values
                        "mean": 0.0,
                        "variance": 0.0,
                        "last_updated": timestamp
                    }
                
                baseline = self.baseline_metrics[metric_name]
                baseline["values"].append(value)
                baseline["last_updated"] = timestamp
                
                # Update statistics
                if len(baseline["values"]) > 1:
                    values = list(baseline["values"])
                    baseline["mean"] = sum(values) / len(values)
                    baseline["variance"] = sum((x - baseline["mean"]) ** 2 for x in values) / len(values)
    
    def detect_anomalies(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in current metrics compared to baseline."""
        anomalies = []
        
        for metric_name, current_value in current_metrics.items():
            if not isinstance(current_value, (int, float)):
                continue
            
            if metric_name not in self.baseline_metrics:
                continue
            
            baseline = self.baseline_metrics[metric_name]
            if len(baseline["values"]) < 10:  # Need enough data
                continue
            
            # Calculate z-score
            if baseline["variance"] > 0:
                std_dev = baseline["variance"] ** 0.5
                z_score = abs((current_value - baseline["mean"]) / std_dev)
                
                if z_score > self.anomaly_threshold:
                    anomalies.append({
                        "metric": metric_name,
                        "current_value": current_value,
                        "baseline_mean": baseline["mean"],
                        "z_score": z_score,
                        "severity": "critical" if z_score > 3.0 else "warning",
                        "description": f"Anomalous {metric_name}: {current_value:.2f} (baseline: {baseline['mean']:.2f})"
                    })
        
        return anomalies


class ThreatIntelligence:
    """Threat intelligence and pattern recognition."""
    
    def __init__(self):
        self.known_attack_patterns = {}
        self.ip_reputation = {}
        self.attack_signatures = set()
        self.logger = structlog.get_logger(__name__)
        
        # Initialize with common attack patterns
        self._initialize_attack_patterns()
    
    def _initialize_attack_patterns(self):
        """Initialize known attack patterns."""
        self.known_attack_patterns = {
            "repeated_injection": {
                "description": "Repeated prompt injection attempts",
                "indicators": ["multiple_injections", "same_source", "short_intervals"],
                "severity": "high",
                "ttl_hours": 24
            },
            "escalation_attempt": {
                "description": "Privilege escalation attempts",
                "indicators": ["system_override", "admin_commands", "role_confusion"],
                "severity": "critical",
                "ttl_hours": 48
            },
            "data_harvesting": {
                "description": "Attempts to extract sensitive data",
                "indicators": ["info_requests", "system_probing", "memory_queries"],
                "severity": "high",
                "ttl_hours": 24
            }
        }
    
    def analyze_event_pattern(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Analyze events for known attack patterns."""
        if not events:
            return {"patterns": [], "risk_score": 0}
        
        detected_patterns = []
        risk_score = 0
        
        # Group events by source
        source_events = defaultdict(list)
        for event in events:
            source_events[event.source].append(event)
        
        # Analyze each source
        for source, source_event_list in source_events.items():
            patterns = self._analyze_source_patterns(source, source_event_list)
            detected_patterns.extend(patterns)
            
            # Calculate risk score contribution
            for pattern in patterns:
                if pattern["severity"] == "critical":
                    risk_score += 30
                elif pattern["severity"] == "high":
                    risk_score += 20
                elif pattern["severity"] == "medium":
                    risk_score += 10
        
        return {
            "patterns": detected_patterns,
            "risk_score": min(100, risk_score),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _analyze_source_patterns(self, source: str, events: List[SecurityEvent]) -> List[Dict[str, Any]]:
        """Analyze patterns for a specific source."""
        patterns = []
        
        # Check for repeated injection attempts
        injection_events = [e for e in events if e.event_type == AlertType.PROMPT_INJECTION]
        if len(injection_events) >= 3:
            time_span = (injection_events[-1].timestamp - injection_events[0].timestamp).total_seconds()
            if time_span < 3600:  # Within 1 hour
                patterns.append({
                    "pattern": "repeated_injection",
                    "source": source,
                    "severity": "high",
                    "event_count": len(injection_events),
                    "time_span_minutes": time_span / 60,
                    "description": f"Repeated injection attempts from {source}"
                })
        
        # Check for escalation attempts
        violation_events = [e for e in events if e.event_type == AlertType.SECURITY_VIOLATION]
        if len(violation_events) >= 2:
            critical_violations = [e for e in violation_events if e.severity == "critical"]
            if critical_violations:
                patterns.append({
                    "pattern": "escalation_attempt",
                    "source": source,
                    "severity": "critical",
                    "event_count": len(critical_violations),
                    "description": f"Privilege escalation attempts from {source}"
                })
        
        return patterns
    
    def update_threat_intelligence(self, events: List[SecurityEvent]):
        """Update threat intelligence based on new events."""
        for event in events:
            # Update IP reputation (if available)
            if event.client_info and "ip" in event.client_info:
                ip = event.client_info["ip"]
                if ip not in self.ip_reputation:
                    self.ip_reputation[ip] = {"score": 0, "events": 0, "last_seen": None}
                
                self.ip_reputation[ip]["events"] += 1
                self.ip_reputation[ip]["last_seen"] = event.timestamp
                
                # Decrease reputation based on severity
                if event.severity == "critical":
                    self.ip_reputation[ip]["score"] -= 10
                elif event.severity == "error":
                    self.ip_reputation[ip]["score"] -= 5
                elif event.severity == "warning":
                    self.ip_reputation[ip]["score"] -= 2
            
            # Extract attack signatures
            if event.details.get("matched_text"):
                signature = hashlib.md5(event.details["matched_text"].encode()).hexdigest()
                self.attack_signatures.add(signature)


class AlertManager:
    """Manages security alerts and notifications."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_alerts: Dict[str, SecurityAlert] = {}
        self.alert_history: List[SecurityAlert] = []
        self.notification_queue = queue.Queue()
        self.logger = structlog.get_logger(__name__)
        
        # Start notification worker
        self.notification_worker = threading.Thread(target=self._process_notifications, daemon=True)
        self.notification_worker.start()
    
    def create_alert(
        self, 
        alert_type: AlertType, 
        severity: AlertSeverity,
        events: List[SecurityEvent],
        title: str,
        message: str
    ) -> SecurityAlert:
        """Create new security alert."""
        
        alert_id = self._generate_alert_id(alert_type, events)
        
        # Check if similar alert already exists
        if alert_id in self.active_alerts:
            existing_alert = self.active_alerts[alert_id]
            return self._update_existing_alert(existing_alert, events)
        
        # Create new alert
        alert = SecurityAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            events=events,
            first_occurrence=min(e.timestamp for e in events),
            last_occurrence=max(e.timestamp for e in events),
            occurrence_count=len(events),
            affected_sources=set(e.source for e in events),
            recommended_actions=self._get_recommended_actions(alert_type, severity),
            is_resolved=False
        )
        
        self.active_alerts[alert_id] = alert
        self._queue_notification(alert)
        
        self.logger.warning(
            "Security alert created",
            alert_id=alert_id,
            alert_type=alert_type.value,
            severity=severity,
            event_count=len(events)
        )
        
        return alert
    
    def _generate_alert_id(self, alert_type: AlertType, events: List[SecurityEvent]) -> str:
        """Generate unique alert ID based on type and event characteristics."""
        sources = sorted(set(e.source for e in events))
        content_hashes = sorted(set(e.content_hash for e in events if e.content_hash))
        
        id_components = [alert_type.value] + sources + content_hashes
        id_string = "|".join(id_components)
        return hashlib.md5(id_string.encode()).hexdigest()[:16]
    
    def _update_existing_alert(self, alert: SecurityAlert, new_events: List[SecurityEvent]) -> SecurityAlert:
        """Update existing alert with new events."""
        updated_events = alert.events + new_events
        updated_alert = SecurityAlert(
            alert_id=alert.alert_id,
            alert_type=alert.alert_type,
            severity=alert.severity,
            title=alert.title,
            message=alert.message,
            events=updated_events,
            first_occurrence=alert.first_occurrence,
            last_occurrence=max(e.timestamp for e in new_events),
            occurrence_count=len(updated_events),
            affected_sources=alert.affected_sources | set(e.source for e in new_events),
            recommended_actions=alert.recommended_actions,
            is_resolved=False
        )
        
        self.active_alerts[alert.alert_id] = updated_alert
        
        # Send update notification for high-severity alerts
        if alert.severity in ["error", "critical"]:
            self._queue_notification(updated_alert, is_update=True)
        
        return updated_alert
    
    def _get_recommended_actions(self, alert_type: AlertType, severity: AlertSeverity) -> List[str]:
        """Get recommended actions based on alert type and severity."""
        actions = []
        
        if alert_type == AlertType.PROMPT_INJECTION:
            actions.extend([
                "Review and strengthen input validation",
                "Update prompt injection detection patterns",
                "Consider blocking problematic sources"
            ])
        elif alert_type == AlertType.SECURITY_VIOLATION:
            actions.extend([
                "Investigate source of violations",
                "Review security guardrails configuration",
                "Consider increasing security level"
            ])
        elif alert_type == AlertType.RESOURCE_ABUSE:
            actions.extend([
                "Implement rate limiting",
                "Monitor resource usage patterns",
                "Consider blocking abusive sources"
            ])
        
        if severity == "critical":
            actions.insert(0, "IMMEDIATE ACTION REQUIRED - Investigate and respond immediately")
        
        return actions
    
    def _queue_notification(self, alert: SecurityAlert, is_update: bool = False):
        """Queue alert for notification."""
        notification = {
            "alert": alert,
            "is_update": is_update,
            "timestamp": datetime.now()
        }
        self.notification_queue.put(notification)
    
    def _process_notifications(self):
        """Process notification queue in background thread."""
        while True:
            try:
                notification = self.notification_queue.get(timeout=1)
                self._send_notification(notification)
                self.notification_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error("Notification processing error", error=str(e))
    
    def _send_notification(self, notification: Dict[str, Any]):
        """Send notification via configured channels."""
        alert = notification["alert"]
        is_update = notification["is_update"]
        
        # Email notification
        if self.config.get("email", {}).get("enabled", False):
            self._send_email_notification(alert, is_update)
        
        # Webhook notification
        if self.config.get("webhook", {}).get("enabled", False):
            self._send_webhook_notification(alert, is_update)
        
        # Log notification
        self.logger.info(
            "Security alert notification sent",
            alert_id=alert.alert_id,
            alert_type=alert.alert_type.value,
            severity=alert.severity,
            is_update=is_update
        )
    
    def _send_email_notification(self, alert: SecurityAlert, is_update: bool):
        """Send email notification for security alert."""
        email_config = self.config.get("email", {})
        
        try:
            msg = MIMEMultipart()
            msg["From"] = email_config["from"]
            msg["To"] = ", ".join(email_config["to"])
            msg["Subject"] = f"{'[UPDATE] ' if is_update else ''}Security Alert: {alert.title}"
            
            body = self._format_alert_email(alert, is_update)
            msg.attach(MIMEText(body, "html"))
            
            # Send email
            with smtplib.SMTP(email_config["smtp_host"], email_config["smtp_port"]) as server:
                if email_config.get("use_tls", True):
                    server.starttls()
                if email_config.get("username") and email_config.get("password"):
                    server.login(email_config["username"], email_config["password"])
                server.send_message(msg)
            
        except Exception as e:
            self.logger.error("Email notification failed", error=str(e))
    
    def _send_webhook_notification(self, alert: SecurityAlert, is_update: bool):
        """Send webhook notification for security alert."""
        webhook_config = self.config.get("webhook", {})
        
        try:
            import requests
            
            payload = {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type.value,
                "severity": alert.severity,
                "title": alert.title,
                "message": alert.message,
                "occurrence_count": alert.occurrence_count,
                "affected_sources": list(alert.affected_sources),
                "is_update": is_update,
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.post(
                webhook_config["url"],
                json=payload,
                headers=webhook_config.get("headers", {}),
                timeout=30
            )
            response.raise_for_status()
            
        except Exception as e:
            self.logger.error("Webhook notification failed", error=str(e))
    
    def _format_alert_email(self, alert: SecurityAlert, is_update: bool) -> str:
        """Format alert information as HTML email."""
        update_text = "<p><strong>This is an update to an existing alert.</strong></p>" if is_update else ""
        
        return f"""
        <html>
        <head></head>
        <body>
        <h2>Security Alert: {alert.title}</h2>
        {update_text}
        
        <h3>Alert Details</h3>
        <ul>
        <li><strong>Alert ID:</strong> {alert.alert_id}</li>
        <li><strong>Type:</strong> {alert.alert_type.value}</li>
        <li><strong>Severity:</strong> {alert.severity}</li>
        <li><strong>Occurrence Count:</strong> {alert.occurrence_count}</li>
        <li><strong>First Occurrence:</strong> {alert.first_occurrence}</li>
        <li><strong>Last Occurrence:</strong> {alert.last_occurrence}</li>
        <li><strong>Affected Sources:</strong> {', '.join(alert.affected_sources)}</li>
        </ul>
        
        <h3>Message</h3>
        <p>{alert.message}</p>
        
        <h3>Recommended Actions</h3>
        <ul>
        {''.join(f'<li>{action}</li>' for action in alert.recommended_actions)}
        </ul>
        
        <h3>Recent Events</h3>
        <ul>
        {''.join(f'<li>{event.timestamp}: {event.message}</li>' for event in alert.events[-5:])}
        </ul>
        </body>
        </html>
        """


class SecurityMonitor:
    """Main security monitoring system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = SecurityMetrics()
        self.anomaly_detector = AnomalyDetector()
        self.threat_intelligence = ThreatIntelligence()
        self.alert_manager = AlertManager(config.get("alerting", {}))
        self.logger = structlog.get_logger(__name__)
        
        # Start monitoring loop
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def record_security_event(
        self,
        event_type: AlertType,
        severity: AlertSeverity,
        source: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        content_hash: Optional[str] = None,
        client_info: Optional[Dict[str, str]] = None
    ):
        """Record a security event."""
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            source=source,
            message=message,
            details=details or {},
            content_hash=content_hash,
            client_info=client_info,
            mitigation_applied=False
        )
        
        self.metrics.add_event(event)
        
        # Check if this event should trigger an alert
        self._evaluate_for_alerts([event])
        
        self.logger.info(
            "Security event recorded",
            event_type=event_type.value,
            severity=severity,
            source=source,
            content_hash=content_hash
        )
    
    def record_detection_result(self, result: DetectionResult, source: str):
        """Record results from security detection analysis."""
        if result.is_malicious:
            self.record_security_event(
                event_type=AlertType.PROMPT_INJECTION,
                severity="critical" if result.risk_score > 80 else "warning",
                source=source,
                message=f"Prompt injection detected (risk score: {result.risk_score})",
                details={
                    "risk_score": result.risk_score,
                    "primary_vector": result.primary_vector.value if result.primary_vector else None,
                    "signal_count": len(result.signals),
                    "detection_summary": result.detection_summary
                },
                content_hash=result.content_hash
            )
    
    def record_security_violation(self, violation: SecurityViolationError, source: str):
        """Record security guardrail violations."""
        severity_map = {
            "critical": "critical",
            "high": "error", 
            "medium": "warning",
            "low": "info"
        }
        
        self.record_security_event(
            event_type=AlertType.SECURITY_VIOLATION,
            severity=severity_map.get(violation.severity, "warning"),
            source=source,
            message=violation.message,
            details={
                "violation_type": violation.violation_type,
                "severity": violation.severity
            }
        )
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background."""
        while self.monitoring_active:
            try:
                # Get current metrics
                current_metrics = self.metrics.get_current_metrics()
                
                # Update anomaly detection baseline
                self.anomaly_detector.update_baseline(current_metrics)
                
                # Check for anomalies
                anomalies = self.anomaly_detector.detect_anomalies(current_metrics)
                
                # Process anomalies
                for anomaly in anomalies:
                    self.record_security_event(
                        event_type=AlertType.ANOMALOUS_BEHAVIOR,
                        severity=anomaly["severity"],
                        source="system_monitor",
                        message=anomaly["description"],
                        details=anomaly
                    )
                
                # Sleep before next check
                time.sleep(self.config.get("monitoring_interval", 300))  # 5 minutes default
                
            except Exception as e:
                self.logger.error("Monitoring loop error", error=str(e))
                time.sleep(60)  # Wait before retrying
    
    def _evaluate_for_alerts(self, new_events: List[SecurityEvent]):
        """Evaluate if new events should trigger alerts."""
        
        # Get recent events for pattern analysis
        recent_events = list(self.metrics.events)[-100:]  # Last 100 events
        
        # Analyze threat patterns
        pattern_analysis = self.threat_intelligence.analyze_event_pattern(recent_events)
        
        # Update threat intelligence
        self.threat_intelligence.update_threat_intelligence(new_events)
        
        # Check for alert conditions
        self._check_alert_conditions(new_events, recent_events, pattern_analysis)
    
    def _check_alert_conditions(
        self, 
        new_events: List[SecurityEvent],
        recent_events: List[SecurityEvent],
        pattern_analysis: Dict[str, Any]
    ):
        """Check various conditions that should trigger alerts."""
        
        # Critical severity events always trigger alerts
        critical_events = [e for e in new_events if e.severity == "critical"]
        if critical_events:
            self.alert_manager.create_alert(
                alert_type=AlertType.SECURITY_VIOLATION,
                severity="critical",
                events=critical_events,
                title="Critical Security Violations Detected",
                message=f"{len(critical_events)} critical security violations detected"
            )
        
        # High-risk pattern detection
        if pattern_analysis["risk_score"] > 70:
            pattern_events = [e for e in recent_events if e.severity in ["error", "critical"]][-10:]
            self.alert_manager.create_alert(
                alert_type=AlertType.REPEATED_ATTACKS,
                severity="error",
                events=pattern_events,
                title="High-Risk Attack Pattern Detected",
                message=f"Attack pattern detected with risk score {pattern_analysis['risk_score']}"
            )
        
        # Threshold breaches
        current_metrics = self.metrics.get_current_metrics()
        if current_metrics["events_per_hour"] > self.config.get("event_rate_threshold", 100):
            threshold_events = recent_events[-20:]
            self.alert_manager.create_alert(
                alert_type=AlertType.THRESHOLD_BREACH,
                severity="warning",
                events=threshold_events,
                title="High Event Rate Detected",
                message=f"Event rate {current_metrics['events_per_hour']:.1f}/hour exceeds threshold"
            )
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data."""
        current_metrics = self.metrics.get_current_metrics()
        active_alerts = list(self.alert_manager.active_alerts.values())
        
        return {
            "metrics": current_metrics,
            "active_alerts": len(active_alerts),
            "critical_alerts": len([a for a in active_alerts if a.severity == "critical"]),
            "recent_patterns": self.threat_intelligence.analyze_event_pattern(list(self.metrics.events)),
            "top_threat_sources": current_metrics.get("top_sources", []),
            "system_health": {
                "monitoring_active": self.monitoring_active,
                "last_update": datetime.now().isoformat()
            }
        }
    
    def shutdown(self):
        """Shutdown monitoring system."""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)


# Example usage and configuration
def create_default_monitor_config() -> Dict[str, Any]:
    """Create default monitoring configuration."""
    return {
        "monitoring_interval": 300,  # 5 minutes
        "event_rate_threshold": 100,  # events per hour
        "alerting": {
            "email": {
                "enabled": False,
                "smtp_host": "localhost",
                "smtp_port": 587,
                "from": "security@example.com",
                "to": ["admin@example.com"],
                "use_tls": True
            },
            "webhook": {
                "enabled": False,
                "url": "https://hooks.slack.com/services/...",
                "headers": {"Content-Type": "application/json"}
            }
        }
    }


def main():
    """Example usage of the security monitoring system."""
    
    # Create monitoring configuration
    config = create_default_monitor_config()
    
    # Initialize security monitor
    monitor = SecurityMonitor(config)
    
    # Simulate some security events
    monitor.record_security_event(
        event_type=AlertType.PROMPT_INJECTION,
        severity="warning",
        source="test_source",
        message="Test prompt injection detected",
        details={"risk_score": 65}
    )
    
    # Get dashboard data
    dashboard = monitor.get_security_dashboard()
    print("Security Dashboard:")
    print(f"- Recent events: {dashboard['metrics']['recent_events']}")
    print(f"- Active alerts: {dashboard['active_alerts']}")
    print(f"- Event rate: {dashboard['metrics']['events_per_hour']:.1f}/hour")
    
    # Shutdown when done
    time.sleep(2)
    monitor.shutdown()


if __name__ == "__main__":
    main()