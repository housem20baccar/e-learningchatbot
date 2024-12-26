# Personalized E-Learning Chatbot

An intelligent, personalized chatbot system for e-learning that adapts to each individual user. Using LLaMA model integration, advanced user behavior analysis, and dynamic content recommendations, the system creates a unique learning experience tailored to each learner's needs, progress, and learning patterns.

## Features

- **Personalized Learning Experience**
  - Individual user profiles and progress tracking
  - Custom learning paths based on user performance
  - Adaptive difficulty levels per user
  - Personal learning history and metrics

- **Smart User Analysis**
  - Individual behavior tracking and analysis
  - Per-user performance metrics
  - Custom progress monitoring
  - Individual learning pace adaptation

- **Technical Features**
  - Interactive chat interface with real-time responses
  - User-specific behavior analysis and clustering
  - Custom content recommendations per user
  - Integration with LLaMA 3.2 model
  - MySQL database for individual user metrics
  - Responsive web interface

## Project Structure

```
e-learningchatbot/
├── app.py              # Main application file with user tracking logic
├── content.json        # Content database for adaptive learning
├── templates/          # Template directory
│   └── interface.html  # Personalized web interface template
└── README.md          # Project documentation
```

## Prerequisites

- Python 3.8+
- MySQL Server
- Ollama with LLaMA 3.2 model
- Flask
- pandas
- scikit-learn
- mysql-connector-python

## Installation

1. Clone the repository
```bash
git clone <your-repository-url>
cd e-learningchatbot
```

2. Install required Python packages
```bash
pip install flask pandas numpy scikit-learn mysql-connector-python ollama
```

3. Set up MySQL database
- Create a new database named 'chatbot'
- Configure database settings in `app.py` if needed

4. Start the application
```bash
python app.py
```

5. Access the web interface at `http://localhost:5000`

## How It Works

The personalized chatbot system:
- Creates and maintains individual user profiles
- Processes user queries using LLaMA 3.2 model
- Tracks individual user metrics including:
  - Query patterns
  - Learning speed
  - Success rates
  - Time spent on topics
  - Challenge completion rates
- Uses Gaussian Mixture Model for user-specific clustering
- Provides personalized content recommendations based on individual progress
- Stores each user's progress and preferences in MySQL database

## Key Features Per User

1. **Individual Progress Tracking**
   - Personal learning history
   - Custom progress metrics
   - Individual performance analysis

2. **Adaptive Learning Path**
   - Personalized content difficulty
   - Custom-paced progression
   - Individual learning recommendations

3. **Performance Analytics**
   - Individual success metrics
   - Personal learning patterns
   - Custom improvement suggestions

4. **User-Specific Interface**
   - Personal chat history
   - Individual recommendations
   - Custom difficulty adjustments
5. **Interfaces and Mysql**
![Screenshot 2024-12-26 205926](https://github.com/user-attachments/assets/e5c61974-abe2-48e1-bead-05867e295b4c)
![Screenshot 2024-12-26 210109](https://github.com/user-attachments/assets/7c6d1050-5b2e-4fcd-8049-69701e3ed998)

