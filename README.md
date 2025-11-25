<<<<<<< HEAD
# crop-price-prediction
The proposed system forecasts crop prices for multiple timeframes—1, 3, 6, and 12 months ahead—while considering the geographical location of the crops. External data sources such as historical market trends, meteorological conditions, and soil quality metrics are incorporated to refine predictions. 
=======
# Crop Price Prediction System

A comprehensive machine learning system for predicting crop prices using ensemble models (LSTM + XGBoost).

## Project Structure

```
crop-price-prediction/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/routes/     # API endpoints
│   │   ├── data/          # Data storage
│   │   └── models/        # ML models
│   ├── train_*.py         # Training scripts
│   └── requirements.txt
└── frontend/              # React frontend
    ├── src/
    │   ├── components/    # UI components
    │   └── pages/        # Page components
    └── package.json
```

## Features

- **Multi-Model Support**: LSTM Neural Networks + XGBoost Gradient Boosting
- **Ensemble Predictions**: Weighted combination of LSTM and XGBoost
- **Real-time Forecasting**: Multi-step price predictions
- **Modern UI**: React frontend with Material-UI components
- **RESTful API**: FastAPI backend with comprehensive endpoints

## Quick Start

### Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## API Endpoints

- `GET /api/v1/latest-features` - Get latest market features
- `POST /api/v1/forecast` - Generate price forecasts
- `POST /api/v1/predict` - Single price prediction
- `GET /api/v1/prediction-status` - Model availability status

## Model Types

1. **XGBoost**: Gradient boosting with 87-100% accuracy
2. **LSTM**: Neural network with temporal dependencies
3. **Ensemble**: Weighted combination of both models

## TODO

- [x] **Backend API Development** - FastAPI with prediction endpoints
- [x] **Frontend UI Development** - React with Material-UI
- [x] **Data Processing Pipeline** - Feature engineering and normalization
- [x] **XGBoost Model Training** - Individual model training with 80/20 split
- [x] **LSTM Model Integration** - Neural network model development
- [x] **Ensemble Model Creation** - Combined LSTM + XGBoost predictions
- [x] **Multi-Model API Support** - Backend supports all three model types
- [x] **Frontend Model Selection** - UI dropdown for model selection
- [x] **LSTM Normalization Fix** - Proper target denormalization
- [x] **Feature Alignment** - API uses correct 11 features for all models

## Current Status

✅ **All models working correctly** - LSTM, XGBoost, and Ensemble predictions are functioning properly
✅ **Frontend integration complete** - Model selection and visualization working
✅ **Backend API stable** - All endpoints returning correct predictions
✅ **Data processing optimized** - Proper feature engineering and normalization

## Next Steps

- [ ] **Model Performance Optimization** - Fine-tune hyperparameters
- [ ] **Additional Crops/Markets** - Expand to more agricultural commodities
- [ ] **Real-time Data Integration** - Live market data feeds
- [ ] **Advanced Analytics** - Price volatility analysis and trends
- [ ] **Mobile App** - React Native mobile application
- [ ] **Deployment** - Production deployment with Docker 
>>>>>>> d61aed8 (updated)
