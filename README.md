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

# Automated Deployment (CI/CD + IaC)

## Overview
This project demonstrates automated deployment of a FastAPI application using AWS EC2, GitHub Actions, and Terraform.

## CI/CD Pipeline
- Triggered on push to main branch
- Uses GitHub Actions
- Connects to EC2 via SSH
- Pulls latest code
- Installs dependencies
- Restarts application automatically

## Infrastructure as Code
Terraform is used to provision AWS resources:
- EC2 instance to host the application
- Security group allowing:
  - Port 22 (SSH)
  - Port 8000 (application access)

## Deployment
- Application runs on EC2
- Updates automatically when code changes

## How It Works
1. Developer pushes code to the main branch
2. GitHub Actions pipeline is triggered
3. Pipeline connects to EC2 via SSH
4. Latest code is pulled
5. Dependencies are installed
6. Application restarts automatically
7. Changes are live immediately

## Automation Proof
- No manual intervention required
- Application updates automatically after push

## Endpoint
http://3.18.111.111:8000/health

## Technologies Used
- Python (FastAPI)
- AWS EC2
- Terraform
- GitHub Actions
- Uvicorn