## Gen AI Powered Customer Insights Recommendation System
App Link:-
https://gen-ai-powered-customer-insights-recommendation-system.streamlit.app/

📌 Overview

This project demonstrates an AI-powered dynamic pricing engine built using Proximal Policy Optimization (PPO) in reinforcement learning. The system simulates real-world market conditions to optimize pricing strategies, aiming to maximize profits while considering factors like demand, competitor pricing, and inventory levels.

🚀 Key Features

Interactive Streamlit Dashboard: A user-friendly interface with sliders to adjust market conditions and visualize the impact on pricing strategies.

Real-Time Price Optimization: Utilizes PPO to predict optimal pricing based on current market dynamics.

Dynamic Profit Visualization: Displays profit trends against various price points, highlighting the predicted optimal price.

Market Condition Randomization: A feature to simulate different market scenarios, enhancing the robustness of the pricing strategy.

Performance Metrics: Tracks and visualizes training rewards over episodes to monitor the learning progress of the model.

🧠 Technical Details

Reinforcement Learning Model: Implemented PPO using TensorFlow/Keras, comprising:

Actor Network: Predicts the optimal pricing action.

Critic Network: Evaluates the value of the current state to guide the actor.

Custom Environment: Developed using Gymnasium to simulate a dynamic pricing scenario with continuous state and action spaces.

Training Loop: The agent learns over 200 episodes, optimizing pricing strategies to maximize profit.

📊 Visuals

Figure: Training rewards over episodes.

Figure: Profit trends against different price points.

🛠️ Installation & Usage

Clone the repository:

git clone https://github.com/debasmita30/Gen-AI-Powered-Customer-Insights-Recommendation-System.git
cd Gen-AI-Powered-Customer-Insights-Recommendation-System


Set up a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py


Access the application in your browser at http://localhost:8501.

🧪 Training the Model

To train the reinforcement learning model:

Navigate to the project directory.

Run the training script:

python train.py


The trained model and training rewards graph will be saved in the models/ directory.

🔧 Technologies Used

Python Libraries: TensorFlow, Gymnasium, NumPy, Matplotlib, Streamlit

Reinforcement Learning Algorithm: Proximal Policy Optimization (PPO)

Development Tools: Jupyter Notebook, Git

📈 Potential Enhancements

Model Evaluation: Implement evaluation metrics such as Average Reward and Win Rate to assess model performance.

Advanced Visualizations: Integrate Plotly for interactive charts and dashboards.

Hyperparameter Tuning: Explore different learning rates and network architectures to improve model performance.

🎯 Alignment with Job Role

This project aligns with roles in data science and machine learning, particularly those focusing on:

Reinforcement Learning: Experience with PPO and continuous action spaces.

Pricing Optimization: Developing strategies to maximize profits in dynamic markets.

AI Deployment: Creating interactive applications using Streamlit for real-time decision-making tools.

