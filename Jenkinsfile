pipeline {
    agent any
    stages {
        stage('Build image') {
            steps {
                // Build the container image using the Dockerfile in the project's root directory
                sh 'docker build ./ -t gcr.io/wave46-mihaiadrian/titanic:cicd'
            }
        }
        stage('Push image') {
            steps {
                sh 'docker push gcr.io/wave46-mihaiadrian/titanic:cicd'
            } 
        }
        stage('get credentials') {
            steps {
                sh 'gcloud container clusters get-credentials titanic-cluster \
                      --region europe-west4 \
                      --project wave46-mihaiadrian'
            }
        }
        stage('apply deployment') {
            steps {
                sh 'kubectl apply -f deployment.yaml'
            }
        }
        stage('apply service') {
            steps {
                sh 'kubectl apply -f service.yaml'
            }
        }
        stage('Deploy on port') {
            steps {
                sh 'kubectl apply -f deployment.yaml'
            }
        }
        stage('Deploy on port') {
            steps {
                sh 'kubectl expose deployment titanic-app --name=titanic-service --type=LoadBalancer --port 80 --target-port 8000'
            }
        }
    }
}
