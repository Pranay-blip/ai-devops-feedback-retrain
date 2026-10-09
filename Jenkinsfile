pipeline {
    agent any

    options {
        disableConcurrentBuilds()
    }

    environment {
        IMAGE     = 'iris-feedback-api'
        CONTAINER = 'iris-feedback-api'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                bat 'python -m venv .venv'
                bat 'call .venv\\Scripts\\python -m pip install -r requirements.txt'
            }
        }

        stage('Unit Tests') {
            steps {
                bat 'call .venv\\Scripts\\python -m pytest -q'
            }
        }

        stage('Retrain and Validate') {
            steps {
                bat 'call .venv\\Scripts\\python -m src.retrain'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t %IMAGE%:%BUILD_NUMBER% -t %IMAGE%:latest .'
            }
        }

        stage('Deploy') {
            steps {
                bat(returnStatus: true, script: 'docker rm -f %CONTAINER%')
                bat 'docker run -d --name %CONTAINER% -p 8000:8000 %IMAGE%:%BUILD_NUMBER%'
            }
        }

        stage('Smoke Test') {
            steps {
                bat 'call .venv\\Scripts\\python scripts\\smoke_test.py'
            }
        }
    }

    post {
        success {
            archiveArtifacts artifacts: 'models/model.joblib, models/model_meta.json', fingerprint: true
            echo "Deployed ${env.IMAGE}:${env.BUILD_NUMBER}"
        }
        failure {
            echo 'Pipeline failed. The live model and running container were not replaced.'
        }
    }
}