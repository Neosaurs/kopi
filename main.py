from flask import Flask, request, jsonify
import pandas as pd
import pickle
import torch
import torch.nn as nn
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables untuk model dan product names
loaded_rbm = None
loaded_product_names = None

# Define RBM class
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.1)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))  # bias hidden
        self.v_bias = nn.Parameter(torch.zeros(n_visible))  # bias visible

    def sample_h(self, v):
        wx = torch.matmul(v, self.W.t()) + self.h_bias
        prob_h = torch.sigmoid(wx)
        return prob_h, torch.bernoulli(prob_h)

    def sample_v(self, h):
        wx = torch.matmul(h, self.W) + self.v_bias
        prob_v = torch.sigmoid(wx)
        return prob_v, torch.bernoulli(prob_v)

    def forward(self, v):
        _, h_sample = self.sample_h(v)
        prob_v_recon, _ = self.sample_v(h_sample)
        return prob_v_recon

    def contrastive_divergence(self, input_data, lr=0.01):
        prob_h, h_sample = self.sample_h(input_data)
        prob_v, v_sample = self.sample_v(h_sample)
        prob_h2, _ = self.sample_h(v_sample)

        self.W.data += lr * (torch.matmul(prob_h.t(), input_data) - torch.matmul(prob_h2.t(), v_sample))
        self.v_bias.data += lr * torch.sum(input_data - v_sample, dim=0)
        self.h_bias.data += lr * torch.sum(prob_h - prob_h2, dim=0)

def load_model():
    """Load RBM model and product names"""
    global loaded_rbm, loaded_product_names
    
    try:
        # Load product names
        with open('product_names.pkl', 'rb') as f:
            loaded_product_names = pickle.load(f)
        
        # Load RBM model
        loaded_rbm = RBM(n_visible=len(loaded_product_names), n_hidden=24)
        loaded_rbm.load_state_dict(torch.load('rbm_model.pth'))
        loaded_rbm.eval()
        
        print(f"Model loaded successfully. Number of products: {len(loaded_product_names)}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': loaded_rbm is not None,
        'products_loaded': loaded_product_names is not None
    })

@app.route('/products', methods=['GET'])
def get_products():
    """Get all available products"""
    if loaded_product_names is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Convert product names to list if it's a pandas Index
        if hasattr(loaded_product_names, 'tolist'):
            products = loaded_product_names.tolist()
        else:
            products = list(loaded_product_names)
        
        return jsonify({
            'products': products,
            'total_products': len(products)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Get product recommendations based on user's bought products"""
    if loaded_rbm is None or loaded_product_names is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        bought_products = data.get('bought_products', [])
        top_n = data.get('top_n', 10)
        exclude_bought = data.get('exclude_bought', True)
        
        if not isinstance(bought_products, list):
            return jsonify({'error': 'bought_products must be a list'}), 400
        
        # Create input vector
        new_user_input_list = [0.0] * len(loaded_product_names)
        found_products = []
        not_found_products = []
        
        for product in bought_products:
            try:
                # Handle pandas Index
                if hasattr(loaded_product_names, 'get_loc'):
                    product_index = loaded_product_names.get_loc(product)
                else:
                    product_index = list(loaded_product_names).index(product)
                
                new_user_input_list[product_index] = 1.0
                found_products.append(product)
            except (KeyError, ValueError):
                not_found_products.append(product)
        
        # Convert to tensor
        new_user_input = torch.tensor([new_user_input_list], dtype=torch.float32)
        
        # Get recommendations
        with torch.no_grad():
            recommendation_scores = loaded_rbm(new_user_input).detach().numpy()[0]
        
        # Create recommendations DataFrame
        if hasattr(loaded_product_names, 'tolist'):
            product_list = loaded_product_names.tolist()
        else:
            product_list = list(loaded_product_names)
        
        recommendations_df = pd.DataFrame({
            'product_name': product_list,
            'recommendation_score': recommendation_scores,
            'already_bought': new_user_input_list
        })
        
        # Sort by recommendation score
        recommendations_df = recommendations_df.sort_values(by='recommendation_score', ascending=False)
        
        # Filter out bought items if requested
        if exclude_bought:
            recommendations_df = recommendations_df[recommendations_df['already_bought'] == 0]
        
        # Get top N recommendations
        top_recommendations = recommendations_df.head(top_n)
        
        # Prepare response
        recommendations = []
        for _, row in top_recommendations.iterrows():
            recommendations.append({
                'product_name': row['product_name'],
                'recommendation_score': float(row['recommendation_score']),
                'already_bought': bool(row['already_bought'])
            })
        
        return jsonify({
            'recommendations': recommendations,
            'input_summary': {
                'bought_products_found': found_products,
                'bought_products_not_found': not_found_products,
                'total_found': len(found_products),
                'total_not_found': len(not_found_products)
            },
            'parameters': {
                'top_n': top_n,
                'exclude_bought': exclude_bought
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommend_batch', methods=['POST'])
def get_batch_recommendations():
    """Get recommendations for multiple users"""
    if loaded_rbm is None or loaded_product_names is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        users_data = data.get('users', [])
        top_n = data.get('top_n', 10)
        exclude_bought = data.get('exclude_bought', True)
        
        if not isinstance(users_data, list):
            return jsonify({'error': 'users must be a list'}), 400
        
        batch_results = []
        
        for user_idx, user_data in enumerate(users_data):
            user_id = user_data.get('user_id', f'user_{user_idx}')
            bought_products = user_data.get('bought_products', [])
            
            # Create input vector for this user
            new_user_input_list = [0.0] * len(loaded_product_names)
            found_products = []
            
            for product in bought_products:
                try:
                    if hasattr(loaded_product_names, 'get_loc'):
                        product_index = loaded_product_names.get_loc(product)
                    else:
                        product_index = list(loaded_product_names).index(product)
                    
                    new_user_input_list[product_index] = 1.0
                    found_products.append(product)
                except (KeyError, ValueError):
                    continue
            
            # Get recommendations for this user
            new_user_input = torch.tensor([new_user_input_list], dtype=torch.float32)
            
            with torch.no_grad():
                recommendation_scores = loaded_rbm(new_user_input).detach().numpy()[0]
            
            # Create recommendations DataFrame
            if hasattr(loaded_product_names, 'tolist'):
                product_list = loaded_product_names.tolist()
            else:
                product_list = list(loaded_product_names)
            
            recommendations_df = pd.DataFrame({
                'product_name': product_list,
                'recommendation_score': recommendation_scores,
                'already_bought': new_user_input_list
            })
            
            recommendations_df = recommendations_df.sort_values(by='recommendation_score', ascending=False)
            
            if exclude_bought:
                recommendations_df = recommendations_df[recommendations_df['already_bought'] == 0]
            
            top_recommendations = recommendations_df.head(top_n)
            
            user_recommendations = []
            for _, row in top_recommendations.iterrows():
                user_recommendations.append({
                    'product_name': row['product_name'],
                    'recommendation_score': float(row['recommendation_score'])
                })
            
            batch_results.append({
                'user_id': user_id,
                'recommendations': user_recommendations,
                'found_products_count': len(found_products)
            })
        
        return jsonify({
            'batch_results': batch_results,
            'parameters': {
                'top_n': top_n,
                'exclude_bought': exclude_bought
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model at startup
    print("Loading RBM model and product names...")
    if load_model():
        print("Model loaded successfully!")
        print("Available endpoints:")
        print("- GET /health - Health check")
        print("- GET /products - Get all products")
        print("- POST /recommend - Get recommendations for single user")
        print("- POST /recommend_batch - Get recommendations for multiple users")
        
        # Start Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please check if model files exist.")
        print("Required files:")
        print("- rbm_model.pth")
        print("- product_names.pkl")