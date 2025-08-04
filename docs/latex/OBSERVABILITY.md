# LaTeX Documentation Pipeline Observability

This document outlines the comprehensive observability implementation for the LaTeX documentation compilation pipeline, integrating with the existing Agent Loop monitoring stack.

## Overview

The LaTeX documentation system now includes production-grade observability with:
- **Prometheus metrics** for build performance and reliability tracking
- **Structured logging** with correlation IDs for distributed tracing
- **Grafana dashboards** for visual monitoring and alerting
- **SLI/SLO definitions** with error budget tracking
- **Automated alerting** for build failures and performance issues

## Architecture Integration

### Existing Stack Integration
The LaTeX monitoring seamlessly integrates with the current observability infrastructure:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ LaTeX Compiler  │────│ Prometheus       │────│ Grafana         │
│ (Port 9091)     │    │ (Port 9090)      │    │ (Port 3000)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌──────────────────┐            │
         └──────────────│ Alertmanager     │────────────┘
                        │ (Notifications)  │
                        └──────────────────┘
```

### Data Flow
1. **LaTeX builds** trigger monitoring script (`latex_monitoring.py`)
2. **Metrics export** to Prometheus on port 9091 during build
3. **Structured logs** written to JSON format with correlation IDs
4. **Prometheus scraping** collects metrics every 30 seconds when active
5. **Grafana visualization** displays real-time and historical data
6. **Alerting rules** trigger notifications for failures or SLO violations

## Metrics Catalog

### Build Performance Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `latex_build_duration_seconds` | Histogram | Time spent in each build stage | `document`, `stage`, `status` |
| `latex_build_success_total` | Counter | Successful builds | `document`, `type` |
| `latex_build_failure_total` | Counter | Failed builds | `document`, `type`, `error_type` |

### Quality Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `latex_warnings_total` | Counter | LaTeX warnings during compilation | `document`, `warning_type` |
| `latex_errors_total` | Counter | LaTeX errors during compilation | `document`, `error_type` |
| `latex_bibtex_references_total` | Gauge | Bibliography references processed | `document` |

### Output Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `latex_pages_generated_total` | Gauge | Pages in final PDF | `document` |
| `latex_output_file_size_bytes` | Gauge | PDF file size | `document` |

### SLI Recording Rules

```prometheus
# Build success rate over 5 minutes
slo:latex_build_success_rate:5m

# Build duration percentiles
slo:latex_build_duration_p95:5m
slo:latex_build_duration_p99:5m

# Error budget consumption over 24 hours
slo:latex_error_budget_remaining:24h
```

## Service Level Objectives (SLOs)

### Availability SLO
- **Target**: 95% build success rate
- **Error Budget**: 5% failure rate over 24 hours
- **Alert Threshold**: < 90% success rate for 5 minutes

### Latency SLO
- **Target**: 95% of builds complete within 120 seconds
- **Alert Threshold**: P95 > 120 seconds for 5 minutes

### Quality SLO
- **Target**: < 0.1 errors per second during builds
- **Alert Threshold**: > 0.1 errors/sec for 3 minutes

## Alerting Strategy

### Critical Alerts (PagerDuty/Immediate)
- **LaTeXCompilationDown**: >50% builds failing for 2 minutes
- **LaTeXNoOutputGenerated**: Build reports success but no PDF created

### Warning Alerts (Email/Slack)
- **LaTeXCompilationSlow**: P95 build time > 120 seconds
- **LaTeXHighErrorRate**: > 0.1 LaTeX errors per second
- **LaTeXBibliographyIssues**: Undefined citations with no bibliography

### Info Alerts (Dashboard/Log)
- **LaTeXHighWarningRate**: > 1 warning per second
- **LaTeXErrorBudgetLow**: < 20% error budget remaining
- **LaTeXOutputTooLarge**: PDF > 50MB generated

## Usage Guide

### Basic Monitoring
```bash
# Compile with monitoring enabled
cd docs/latex
make hrm-paper-monitored

# Check metrics endpoint (during build)
curl http://localhost:9091/metrics
```

### Advanced Monitoring
```bash
# Manual monitoring with custom correlation ID
python3 scripts/latex_monitoring.py \
  --document hierarchical_reasoning_model \
  --target hrm-paper \
  --correlation-id "release-v1.2.3" \
  --metrics-port 9091
```

### Log Analysis
```bash
# Filter logs by correlation ID
grep "correlation_id.*latex-123456" /var/log/latex/*.log

# Search for build failures
grep '"level":"error"' /var/log/latex/*.log | jq '.document, .error'
```

## Dashboard Access

### Grafana Dashboard: "LaTeX Documentation Pipeline"
**URL**: http://localhost:3000/d/latex-docs/latex-documentation-pipeline

**Key Panels**:
1. **Build Success Rate**: Real-time success percentage
2. **Build Duration P95**: Compilation performance
3. **Build Rate by Document**: Activity levels per document
4. **Compilation Duration by Stage**: Stage-wise performance breakdown
5. **LaTeX Warnings & Errors**: Quality metrics
6. **Document Output Metrics**: Generated content statistics
7. **Error Budget Remaining**: SLO compliance tracking
8. **Recent Build Status**: Structured log viewer

### Template Variables
- **Document Filter**: Select specific documents to monitor
- **Time Range**: Standard Grafana time controls

## Correlation ID Strategy

### Automatic Generation
- **Format**: `latex-{timestamp}-{username}`
- **Propagation**: Included in all logs and traces
- **Use Cases**: Cross-service debugging, build tracking

### Manual Override
```bash
# Set custom correlation ID for release builds
CORRELATION_ID="release-v1.2.3-hrm-paper" make hrm-paper-monitored
```

### Distributed Tracing
Integration with OpenTelemetry for full request tracing:
```python
# Example: Trace LaTeX build in CI/CD pipeline
with tracer.start_as_current_span("documentation.build") as span:
    span.set_attribute("document", "hierarchical_reasoning_model")
    span.set_attribute("correlation_id", correlation_id)
    # LaTeX compilation happens here
```

## Performance Considerations

### Monitoring Overhead
- **CPU Impact**: < 1% during builds
- **Memory Usage**: ~50MB for monitoring process
- **Network**: ~1KB/second metrics export during builds
- **Storage**: ~100KB per build for structured logs

### Optimization Tips
1. **Selective Monitoring**: Only enable for important documents
2. **Log Rotation**: Configure daily rotation for structured logs
3. **Metrics Retention**: Set appropriate Prometheus retention (30 days)
4. **Sampling**: Use correlation ID sampling in high-volume scenarios

## Integration Examples

### CI/CD Integration
```yaml
# GitHub Actions / Jenkins
- name: Build Documentation with Monitoring
  run: |
    export CORRELATION_ID="${GITHUB_RUN_ID}-docs"
    cd docs/latex
    make hrm-paper-monitored
    
    # Check if build succeeded via metrics
    if curl -s http://localhost:9091/metrics | grep -q "latex_build_success_total.*1"; then
      echo "Build succeeded with monitoring"
    else
      echo "Build failed - check Grafana dashboard"
      exit 1
    fi
```

### Automated Quality Gates
```bash
#!/bin/bash
# quality-gate.sh - Fail builds with too many warnings

BUILD_WARNINGS=$(curl -s http://localhost:9091/metrics | \
  grep "latex_warnings_total" | \
  awk '{sum += $2} END {print sum}')

if [ "$BUILD_WARNINGS" -gt 10 ]; then
  echo "Quality gate failed: $BUILD_WARNINGS warnings (max: 10)"
  exit 1
fi
```

## Troubleshooting

### Common Issues

#### 1. Metrics Not Appearing
```bash
# Check if monitoring script is running
ps aux | grep latex_monitoring.py

# Verify Prometheus target
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.job=="latex-builds")'
```

#### 2. High Memory Usage
```bash
# Monitor process resources
python3 -c "
import psutil
for p in psutil.process_iter(['pid', 'name', 'memory_info']):
    if 'latex_monitoring' in p.info['name']:
        print(f\"PID: {p.info['pid']}, Memory: {p.info['memory_info'].rss / 1024 / 1024:.1f}MB\")
"
```

#### 3. Build Timeouts
- Check LaTeX compilation logs for stuck processes
- Verify disk space for temporary files
- Monitor system resources during builds

### Log Levels
```bash
# Enable debug logging
export STRUCTLOG_LEVEL=DEBUG
make hrm-paper-monitored

# Filter by log level
grep '"level":"error"' /var/log/latex/*.log
```

## Security Considerations

### Data Sanitization
- No sensitive data in LaTeX logs
- File paths sanitized in metrics labels
- Correlation IDs are non-sensitive identifiers

### Access Control
- Metrics endpoint (9091) should be internal-only
- Grafana dashboards require authentication
- Log files have appropriate file permissions (640)

### Privacy
- No document content exposed in metrics
- Only metadata and performance statistics tracked
- Compliance with documentation access policies

## Future Enhancements

### Planned Features
- [ ] Multi-document parallel build monitoring
- [ ] Integration with Git commit tracking
- [ ] Automated performance regression detection
- [ ] Cost tracking for large document compilation
- [ ] Integration with document review workflows

### Advanced Metrics
- [ ] LaTeX package usage statistics
- [ ] Figure/table complexity metrics
- [ ] Citation network analysis
- [ ] Document readability scoring

This observability implementation ensures that the LaTeX documentation pipeline operates with the same reliability and visibility standards as the core Agent Loop ML platform.