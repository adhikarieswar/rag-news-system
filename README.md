# 📰 RAG News System

## AI-Powered Semantic News Search using Retrieval Augmented Generation

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-orange.svg)](https://github.com/features/actions)

---

## 🎯 Overview

**RAG News System** is a production-ready Retrieval Augmented Generation (RAG) application that allows users to ask questions in natural language and get AI-generated answers based on real news articles.

Instead of traditional keyword search, this system understands the **meaning** of your question and finds the most relevant news articles using semantic search, then uses a Large Language Model (LLM) to generate a precise answer with source attribution.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Semantic Search** | Finds articles by meaning, not just keywords |
| 🤖 **AI-Generated Answers** | Uses Groq LLM to generate natural language responses |
| 📚 **Source Attribution** | Every answer includes the source news articles |
| 🚀 **FastAPI Backend** | High-performance async API with automatic docs |
| 🐳 **Docker Support** | Containerized for easy deployment |
| ⚙️ **CI/CD Pipeline** | Automated testing and building with GitHub Actions |
| ✅ **Unit Tests** | 15+ tests covering core functionality |

---

## 🏗️ Architecture

```text
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   User      │────▶│   FastAPI   │────▶│  ChromaDB   │────▶│   Groq LLM  │
│  Question   │     │  Backend    │     │  (Vectors)  │     │  (Answer)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │                    │                    │
                           ▼                    ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
                    │  Sentence   │     │   ETL       │     │   Source    │
                    │Transformer  │     │  Pipeline   │     │ Attribution │
                    └─────────────┘     └─────────────┘     └─────────────┘
