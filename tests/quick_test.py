from src.serve.app import app
import json

def test_quick():
    """Quick test to verify Flask app functionality"""
    with app.test_client() as client:
        # Test health endpoint
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        assert 'timestamp' in data
        assert 'model_loaded' in data
        print("Health endpoint test passed!")

        # Test metadata endpoint
        response = client.get('/metadata')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'model_name' in data
        assert 'model_version' in data
        print("Metadata endpoint test passed!")

if __name__ == "__main__":
    test_quick() 