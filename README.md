# Personal AI Assistant

A transparent AI assistant that learns from user activities and can automate tasks on behalf of the user.

## Features

- **Activity Monitoring**: Captures and analyzes user interactions across applications
- **Behavior Modeling**: Creates a digital profile of the user's behavior patterns using RNN/Transformer models
- **Task Automation**: Executes tasks based on learned patterns with prebuilt recipes and a no-code builder
- **Communication Clone**: Communicates on behalf of the user in their style via fine-tuned NLP models
- **Privacy Controls**: Transparent data handling with granular user control through a privacy dashboard

## Architecture

### Frontend
- React.js with Material-UI components
- Component-driven UI architecture
- Continuous deployment via Netlify

### Backend
- FastAPI (Python) for async task handling and REST endpoints
- Microservices architecture for scalability
- Containerized via Docker and deployed to Heroku

### ML/AI Layer
- Behavior modeling using TensorFlow/PyTorch
- Sequence modeling (RNN/Transformer) for pattern recognition
- Event-driven pipelines with Redis queues

### Data Storage
- SQLite (development), PostgreSQL (production)
- Data Lake architecture for raw activity logs
- Feature Store for cleaned training features

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-assistant.git
cd ai-assistant

# Backend setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cd backend
uvicorn main:app --reload

# Frontend setup (in a new terminal)
cd frontend
npm install
npm start
```

## Deployment

### Frontend (Netlify)
```bash
cd frontend
npm run build
# Deploy using Netlify CLI or GitHub integration
```

### Backend (Heroku)
```bash
# Add Procfile for Heroku in project root
# web: cd backend && uvicorn main:app --host=0.0.0.0 --port=${PORT:-8000}

# Deploy using Heroku CLI
heroku create your-ai-assistant
git push heroku main
```

## Development Status

Currently in prototype phase with the following components implemented:
- Core activity monitoring services
- Basic behavior modeling with sequence prediction
- Task automation framework with event triggers
- Privacy controls and user settings
- Frontend dashboard for visualizing assistant learning

## Future Development
- Enhanced communication cloning with style transfer
- Advanced automation workflows with conditional logic
- Integration with more third-party services and APIs
- Mobile application companion
- On-device model deployment for enhanced privacy
