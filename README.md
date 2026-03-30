# M4 - LLM Integration (n8n)

## Overview
This project demonstrates integration of a Large Language Model (LLM) using n8n and OpenAI API.

## Features
- Webhook accepts user input
- Sends request to OpenAI (LLM)
- Returns AI-generated response

## Workflow
Webhook → HTTP Request → Respond to Webhook

## How to Run
1. Import workflow.json into n8n
2. Add your OpenAI API key in HTTP node
3. Run the workflow
4. Send POST request:


# Automated Deployment (CI/CD + IaC)

## Overview
This project demonstrates automated deployment of a FastAPI application using AWS EC2, GitHub Actions, and Terraform.

## CI/CD Pipeline
- Triggered on push to main branch
- Uses GitHub Actions
- Connects to EC2 via SSH
- Pulls latest code
- Restarts application automatically

## Infrastructure as Code
- Terraform defines EC2 and security group

## Deployment
- Application runs on EC2
- Updates automatically when code changes

## Endpoint
http://3.139.100.83:8000/health