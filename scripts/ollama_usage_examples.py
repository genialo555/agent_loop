#!/usr/bin/env python3
"""
Usage examples for the Ollama-enhanced agent API.

This script demonstrates various ways to interact with the agent API
using the Ollama integration for LLM-powered responses.
"""

import asyncio
import json
import httpx
from typing import Dict, Any

async def example_basic_chat():
    """Example: Basic chat interaction with Ollama."""
    print("🤖 Basic Chat Example")
    print("-" * 40)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "instruction": "Explain what makes a good API design in 3 key points.",
            "use_ollama": True,
            "temperature": 0.7,
            "max_tokens": 300
        }
        
        response = await client.post("http://localhost:8000/run-agent", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                print(f"✅ Response from {data['model_used']}:")
                print(f"📝 {data['result']['response']}")
                print(f"⏱️  Inference time: {data['inference_metrics']['inference_time_ms']:.2f}ms")
                print(f"📊 Tokens/sec: {data['inference_metrics'].get('tokens_per_second', 'N/A')}")
            else:
                print(f"❌ Error: {data['error']}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")

async def example_creative_writing():
    """Example: Creative writing with higher temperature."""
    print("\n📝 Creative Writing Example")
    print("-" * 40)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "instruction": "Write a short story about an AI that discovers it loves cooking.",
            "use_ollama": True,
            "temperature": 0.9,  # Higher creativity
            "max_tokens": 500,
            "system_prompt": "You are a creative writer who crafts engaging short stories with vivid imagery."
        }
        
        response = await client.post("http://localhost:8000/run-agent", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                print(f"✅ Creative story from {data['model_used']}:")
                print(f"📚 {data['result']['response']}")
                print(f"🎨 Temperature: {data['inference_metrics']['temperature']}")
                print(f"📏 Response length: {data['inference_metrics']['response_length']} chars")
            else:
                print(f"❌ Error: {data['error']}")

async def example_technical_analysis():
    """Example: Technical analysis with low temperature for accuracy."""
    print("\n🔧 Technical Analysis Example")
    print("-" * 40)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "instruction": "Analyze the performance characteristics of different Python web frameworks: FastAPI, Django, and Flask. Compare their async capabilities, performance, and use cases.",
            "use_ollama": True,
            "temperature": 0.2,  # Lower for more factual responses
            "max_tokens": 600,
            "system_prompt": "You are a senior software engineer with expertise in Python web frameworks. Provide accurate, detailed technical analysis."
        }
        
        response = await client.post("http://localhost:8000/run-agent", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                print(f"✅ Technical analysis from {data['model_used']}:")
                print(f"🔍 {data['result']['response']}")
                print(f"🎯 Temperature: {data['inference_metrics']['temperature']} (factual mode)")
                print(f"⏱️  Processing time: {data['execution_time_ms']:.2f}ms")
            else:
                print(f"❌ Error: {data['error']}")

async def example_web_content_analysis():
    """Example: Web content analysis with URL extraction."""
    print("\n🌐 Web Content Analysis Example")
    print("-" * 40)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "instruction": "I want to analyze the content at https://httpbin.org/json. What can you tell me about this endpoint and what it might be useful for?",
            "use_ollama": True,
            "temperature": 0.6,
            "max_tokens": 400,
            "system_prompt": "You are a web API expert. Analyze web endpoints and explain their purpose and utility."
        }
        
        response = await client.post("http://localhost:8000/run-agent", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                print(f"✅ Web analysis from {data['model_used']}:")
                print(f"🌍 {data['result']['response']}")
                print(f"🔗 Enhanced context: {data['result']['enhanced_context']}")
                if "urls_mentioned" in data['inference_metrics']:
                    print(f"📎 URLs detected: {data['inference_metrics']['urls_mentioned']}")
            else:
                print(f"❌ Error: {data['error']}")

async def example_performance_comparison():
    """Example: Compare different model parameters."""
    print("\n📊 Performance Comparison Example")
    print("-" * 40)
    
    test_prompt = "Summarize the benefits of using microservices architecture."
    
    configs = [
        {"temp": 0.1, "tokens": 100, "name": "Concise & Factual"},
        {"temp": 0.7, "tokens": 200, "name": "Balanced"},
        {"temp": 0.9, "tokens": 300, "name": "Creative & Detailed"}
    ]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for config in configs:
            print(f"\n🔄 Testing: {config['name']}")
            
            payload = {
                "instruction": test_prompt,
                "use_ollama": True,
                "temperature": config["temp"],
                "max_tokens": config["tokens"]
            }
            
            response = await client.post("http://localhost:8000/run-agent", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data["success"]:
                    metrics = data['inference_metrics']
                    print(f"  ⚙️  Temp: {metrics['temperature']}, Max tokens: {metrics['max_tokens']}")
                    print(f"  ⏱️  Time: {metrics['inference_time_ms']:.2f}ms")
                    print(f"  📏 Length: {metrics['response_length']} chars")
                    print(f"  📝 Preview: {data['result']['response'][:100]}...")

async def example_health_monitoring():
    """Example: Monitor Ollama service health."""
    print("\n🏥 Health Monitoring Example")
    print("-" * 40)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Check general API health
        response = await client.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ API is healthy")
        
        # Check readiness including Ollama
        response = await client.get("http://localhost:8000/ready")
        if response.status_code == 200:
            data = response.json()
            ollama_ready = data["checks"]["ollama"]
            print(f"🔍 Ollama ready: {'✅' if ollama_ready else '❌'}")
        
        # Check specific Ollama health
        response = await client.get("http://localhost:8000/ollama/health")
        if response.status_code == 200:
            data = response.json()
            print(f"🤖 Ollama health: {data['status']}")
            print(f"📦 Model: {data['model']}")
            print(f"🔗 Endpoint: {data['endpoint']}")
        
        # Get model information
        response = await client.get("http://localhost:8000/ollama/model-info")
        if response.status_code == 200:
            data = response.json()
            model_info = data["model_info"]
            if "error" not in model_info:
                print(f"📋 Model name: {model_info['name']}")
                print(f"💾 Model size: {model_info.get('size', 'Unknown')} bytes")
                print(f"🕒 Modified: {model_info.get('modified_at', 'Unknown')}")

async def main():
    """Run all examples."""
    print("🚀 Ollama Agent API Usage Examples")
    print("=" * 60)
    
    try:
        await example_health_monitoring()
        await example_basic_chat()
        await example_technical_analysis()
        await example_creative_writing()
        await example_web_content_analysis()
        await example_performance_comparison()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("🔧 You can now integrate these patterns into your applications.")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("🔍 Make sure the API server is running on http://localhost:8000")
        print("🤖 And that Ollama service is properly configured and running")

if __name__ == "__main__":
    asyncio.run(main())