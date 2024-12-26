from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import ollama
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union
import mysql.connector
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ELearningChatbot:
    def __init__(self):
            self.app = Flask(__name__)
            self.setup_routes()
            
            # Simplified database connection configuration
            self.db_config = {
                'host': "127.0.0.1",
                'port': 3306,
                'user': "root",
                'password': "",
                'database': "chatbot",
                'connection_timeout': 5  # Connection timeout in seconds
            }
            
            # Initialize database connection
            self.cnx = None
            self.cursor = None
            self.connect_to_database()
            
            # Load application data
            self.load_data()
            
            # Model and configuration settings
            self.LLAMA_MODEL = 'llama3.2:3b'  
            self.MIN_CLUSTERS = 3

    def connect_to_database(self):
        """Establish a connection to the MySQL database with robust error handling."""
        try:
            self.cnx = mysql.connector.connect(**self.db_config)
            self.cursor = self.cnx.cursor(dictionary=True)
            logger.info("Successfully connected to the database.")
        except mysql.connector.Error as err:
            logger.error(f"Database connection error: {err}")
            self.cnx = None
            self.cursor = None
            raise

    def ensure_database_tables(self):
        """Ensure required tables exist, creating them if they don't."""
        tables_to_check = [
            """CREATE TABLE IF NOT EXISTS usermetrics (
                user_id VARCHAR(255) PRIMARY KEY,
                queries_count INT DEFAULT 0,
                quiz_score FLOAT DEFAULT 0,
                time_spent FLOAT DEFAULT 0,
                wmc FLOAT DEFAULT 0,
                Challenge_Success_Rate FLOAT DEFAULT 0,
                Learning_Frequency FLOAT DEFAULT 0,
                cluster INT DEFAULT -1
            )"""
        ]
        
        try:
            for table_query in tables_to_check:
                self.cursor.execute(table_query)
            self.cnx.commit()
            logger.info("Database tables verified/created successfully.")
        except mysql.connector.Error as err:
            logger.error(f"Error creating tables: {err}")
            raise

    def load_data(self) -> None:
        """Load content and user data from sources."""
        self.ensure_database_tables()
        
        # Load content database
        try:
            with open('content.json', 'r') as f:
                self.content_db = json.load(f)
        except FileNotFoundError:
            logger.warning("content.json not found. Initializing empty content database.")
            self.content_db = {"clusters": {}}

        # Load user metrics from database
        try:
            self.cursor.execute("SELECT * FROM usermetrics")
            results = self.cursor.fetchall()
            
            if results:
                self.user_data = pd.DataFrame(results)
            else:
                logger.warning("usermetrics table is empty.")
                self.user_data = pd.DataFrame(columns=[
                    'user_id', 'queries_count', 'quiz_score', 'time_spent', 
                    'wmc', 'Challenge_Success_Rate', 'Learning_Frequency', 'cluster'
                ])
        except mysql.connector.Error as e:
            logger.error(f"Error loading user metrics: {e}")
            self.user_data = pd.DataFrame(columns=[
                'user_id', 'queries_count', 'quiz_score', 'time_spent', 
                'wmc', 'Challenge_Success_Rate', 'Learning_Frequency', 'cluster'
            ])

    def save_user_data(self, user_data):
        """Save or update user data in the database."""
        try:
            # Use INSERT ... ON DUPLICATE KEY UPDATE for upsert functionality
            insert_query = """
            INSERT INTO usermetrics 
            (user_id, queries_count, quiz_score, time_spent, wmc, 
             Challenge_Success_Rate, Learning_Frequency, cluster) 
            VALUES (%(user_id)s, %(queries_count)s, %(quiz_score)s, 
                    %(time_spent)s, %(wmc)s, %(Challenge_Success_Rate)s, 
                    %(Learning_Frequency)s, %(cluster)s)
            ON DUPLICATE KEY UPDATE 
            queries_count = %(queries_count)s,
            quiz_score = %(quiz_score)s,
            time_spent = %(time_spent)s,
            wmc = %(wmc)s,
            Challenge_Success_Rate = %(Challenge_Success_Rate)s,
            Learning_Frequency = %(Learning_Frequency)s,
            cluster = %(cluster)s
            """
            
            # Convert DataFrame to list of dictionaries for bulk insert
            data_to_insert = user_data.to_dict('records')
            self.cursor.executemany(insert_query, data_to_insert)
            self.cnx.commit()
            logger.info(f"Saved {len(data_to_insert)} user records.")
        except mysql.connector.Error as e:
            logger.error(f"Error saving user data: {e}")
            self.cnx.rollback()

    def process_input(self, user_query: str) -> str:
        """Process user input using Ollama."""
        try:
            response = ollama.chat(model=self.LLAMA_MODEL, messages=[{
                'role': 'user',
                'content': user_query
            }])
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return "I apologize, but I'm having trouble processing your request right now."
        
    def recommend_content(self, user_cluster: int) -> List[str]:
        """Recommend content based on user cluster."""
        user_level = self.cluster_mapping.get(user_cluster, "Beginner")
        logger.info(f"Determined user level: {user_level}")
        cluster_mapping = {
            "Beginner": ['0', '6', '9','17','18','19'],
            "Intermediate": ['1', '3', '7'],
            "Advanced": ['2', '4', '5', '15']
        }
        recommended_cluster_ids = cluster_mapping.get(user_level, ['0'])
        recommendations = []
        for cluster_id in recommended_cluster_ids:
            cluster_content = self.content_db['clusters'].get(cluster_id, [])
            recommendations.extend(cluster_content)
        return recommendations

    def update_user_cluster(self) -> None:
        """Update user clusters using Gaussian Mixture Model with expanded features."""
        if len(self.user_data) < self.MIN_CLUSTERS:
            logger.warning("Not enough data for meaningful clustering")
            return
        
        features_for_clustering = [
            'queries_count', 
            'quiz_score', 
            'time_spent', 
            'wmc', 
            'Challenge_Success_Rate', 
            'Learning_Frequency'
        ]
        
        # Verify data completeness
        missing_features = [f for f in features_for_clustering if f not in self.user_data.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            return
        
        # Check for sufficient non-null data
        feature_data = self.user_data[features_for_clustering]
        
        # Replace any NaN values with column means to prevent clustering issues
        feature_data = feature_data.fillna(feature_data.mean())
        
        # Normalize features using Min-Max scaling
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(feature_data)
        
        logger.info(f"Normalized Features (6 dimensions):\n{normalized_features}")

        # Apply Gaussian Mixture Model (GMM)
        try:
            gmm = GaussianMixture(n_components=3, random_state=42)
            self.user_data['cluster'] = gmm.fit_predict(normalized_features)            
            cluster_means = self.user_data.groupby('cluster')[features_for_clustering].mean()
            logger.info(f"Cluster Means:\n{cluster_means}")
            
            # Sort clusters based on a comprehensive performance metric
            # Using a weighted combination of features
            performance_metric = cluster_means.apply(
                lambda row: (
                    row['quiz_score'] * 0.3 + 
                    row['Challenge_Success_Rate'] * 0.2 + 
                    row['Learning_Frequency'] * 0.2 + 
                    row['wmc'] * 0.1 + 
                    row['time_spent'] * 0.1 + 
                    row['queries_count'] * 0.1
                ), 
                axis=1
            )
            
            # Map clusters to levels based on performance
            sorted_clusters = performance_metric.sort_values().index.tolist()
            self.cluster_mapping = {
                sorted_clusters[0]: "Beginner",
                sorted_clusters[1]: "Intermediate",
                sorted_clusters[2]: "Advanced",
            }
            
            logger.info(f"Cluster Mapping: {self.cluster_mapping}")
            logger.info(f"Performance Metric: {performance_metric}")
            
            # Save updated user data to database
            self.save_user_data(self.user_data)
            logger.info("Updated user clusters saved to database")
        
        except Exception as e:
            logger.error(f"Clustering error: {e}")
            raise

    def setup_routes(self) -> None:
        """Set up Flask routes."""
        self.app.route('/')(self.home)
        self.app.route('/chat', methods=['POST'])(self.chat)

    def home(self):
        """Render home page."""
        return render_template('interface.html')

    def chat(self):
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            user_id = data.get('user_id')
            user_query = data.get('query')

            if not all([user_id, user_query]):
                return jsonify({"error": "Missing required parameters"}), 400
            
            # Check if user exists, if not create a new record
            existing_user_query = "SELECT * FROM usermetrics WHERE user_id = %s"
            self.cursor.execute(existing_user_query, (user_id,))
            existing_user = self.cursor.fetchone()

            if not existing_user:
                # Comprehensive insert query for all features
                insert_query = """
                INSERT INTO usermetrics 
                (user_id, queries_count, quiz_score, time_spent, 
                wmc, Challenge_Success_Rate, Learning_Frequency, cluster) 
                VALUES (%s, 1, 0, 0, 0, 0, 0, -1)
                """
                self.cursor.execute(insert_query, (user_id,))
                self.cnx.commit()
                
                # Create new user DataFrame entry with all features
                new_user_data = pd.DataFrame({
                    'user_id': [user_id],
                    'queries_count': [1],
                    'quiz_score': [0],
                    'time_spent': [0],
                    'wmc': [0],
                    'Challenge_Success_Rate': [0],
                    'Learning_Frequency': [0],
                    'cluster': [-1]
                })
                self.user_data = pd.concat([self.user_data, new_user_data], ignore_index=True)
            else:
                # Update queries count
                update_query = """
                UPDATE usermetrics 
                SET queries_count = queries_count + 1 
                WHERE user_id = %s
                """
                self.cursor.execute(update_query, (user_id,))
                self.cnx.commit()
                
                # Update queries count in the DataFrame
                self.user_data.loc[self.user_data['user_id'] == user_id, 'queries_count'] += 1

            # Update user clustering
            self.update_user_cluster()
            
            # Get user cluster
            user_cluster = self.user_data.loc[self.user_data['user_id'] == user_id, 'cluster'].iloc[0]
            
            # Process input and get recommendations
            llama_response = self.process_input(user_query)
            content_recommendations = self.recommend_content(user_cluster)

            logger.info(f"User Cluster: {user_cluster}")
            logger.info(f"Content Recommendations: {content_recommendations}")

            return jsonify({
                "response": llama_response,
                "cluster_recommendations": content_recommendations
            })

        except Exception as e:
            logger.error(f"Error in chat endpoint: {e}", exc_info=True)
            return jsonify({"error": "Internal server error"}), 500
    
    def __del__(self):
        """Close database connection when object is destroyed."""
        try:
            if self.cnx and self.cnx.is_connected():
                self.cursor.close()
                self.cnx.close()
                logger.info("Database connection closed.")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

def main():
    chatbot = ELearningChatbot()
    chatbot.app.run(debug=True)

if __name__ == "__main__":
    main()