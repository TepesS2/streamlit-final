# 🚭 Smoking & Risk Factors Dashboard

An interactive Streamlit dashboard for exploring smoking patterns and associated risk factors through comprehensive data visualization.

## 📋 Project Overview

This dashboard analyzes a comprehensive dataset on smoking habits and related risk factors, providing insights into:
- Smoking prevalence across different demographics
- Health metrics and their correlation with smoking status
- Risk factors and lifestyle patterns
- Interactive data exploration capabilities

## 🚀 Features

- **📊 6+ Interactive Visualizations**: Including pie charts, histograms, box plots, correlation matrices, and custom scatter plots
- **🔍 Multi-page Navigation**: Organized into logical sections for easy exploration
- **⚙️ Dynamic Filters**: Real-time filtering by age, gender, smoking status, and other variables
- **📱 Responsive Design**: Works on desktop and mobile devices
- **🎨 Modern UI**: Clean, professional interface with custom styling

## 📁 Project Structure

```
trabalho/
├── app.py                          # Main application file
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
├── data/                          # Data storage directory
├── pages/                         # Additional Streamlit pages
│   ├── 1_Advanced_Analytics.py    # Advanced analytics page
│   └── 2_Data_Export.py          # Data export functionality
└── utils/                         # Utility functions
    ├── data_processing.py         # Data cleaning and processing
    └── visualizations.py          # Chart creation functions
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd trabalho
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the dashboard**
   Open your web browser and navigate to `http://localhost:8501`

## 📊 Dashboard Pages

### 🏠 Home
- Project overview and documentation
- Dataset information and sample data
- Navigation instructions

### 📈 Overview Analysis
- General smoking statistics
- Age and demographic distributions
- Key performance indicators

### 🔍 Demographic Analysis
- Smoking patterns by gender and age groups
- Educational level correlations
- Population segment analysis

### 🏥 Health Metrics
- BMI distributions by smoking status
- Health condition correlations
- Medical risk assessments

### 🎯 Risk Factors
- Lifestyle factor analysis
- Risk correlation matrices
- Predictive insights

### 📊 Interactive Explorer
- Customizable scatter plots
- Dynamic variable selection
- Real-time filtering

## 🔧 How Filters Work

The dashboard includes several interactive filters that affect all visualizations:

- **Age Range Slider**: Filter participants by age range
- **Gender Selection**: Focus on specific gender groups
- **Smoking Status**: Include/exclude different smoking categories
- **Real-time Updates**: All charts update automatically when filters change

## 📈 Key Visualizations

1. **Smoking Status Pie Chart** (Interactive)
2. **Age Distribution Histogram**
3. **Gender vs Smoking Bar Chart**
4. **BMI Box Plots by Smoking Status** (Interactive)
5. **Correlation Heatmap**
6. **Custom Scatter Plot Explorer** (Fully Interactive)

## 🌐 Deployment

### Streamlit Cloud Deployment

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial dashboard setup"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select the `app.py` file
   - Deploy!

### Alternative Deployment Options
- **Heroku**: Use the included `requirements.txt`
- **AWS/GCP**: Deploy as a containerized application
- **Local Network**: Use `streamlit run app.py --server.address 0.0.0.0`

## 📦 Dependencies

- **streamlit** >= 1.28.0: Web framework for the dashboard
- **pandas** >= 2.0.0: Data manipulation and analysis
- **plotly** >= 5.15.0: Interactive visualizations
- **numpy** >= 1.24.0: Numerical computations
- **kagglehub** >= 0.2.0: Dataset download functionality

## 📊 Dataset Information

**Source**: [Smoking and Other Risk Factors Dataset](https://www.kaggle.com/datasets/khushikyad001/smoking-and-other-risk-factors-dataset)

**Size**: 2000+ records meeting project requirements

**Features**: Demographic information, health metrics, lifestyle factors, and smoking status

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

Created as part of an interactive data visualization project.

## 🆘 Support

If you encounter any issues:

1. Check that all dependencies are installed correctly
2. Ensure you have a stable internet connection for dataset download
3. Verify Python version compatibility
4. Check the terminal output for specific error messages

For additional help, please create an issue in the GitHub repository.

---

**Happy Data Exploring! 📊🚭**
