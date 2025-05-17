import matplotlib.pyplot as plt
import json
import os

def plot_model_performance():
    metrics = {
        'MSE': 0.0990,
        'R2': 0.8481,
        'Accuracy': 0.85
    }
    names = list(metrics.keys())
    values = list(metrics.values())
    plt.figure(figsize=(5, 3))
    bars = plt.bar(names, values, color=['#4F81BD', '#C0504D', '#9BBB59'])
    plt.title('Model Performance Metrics')
    plt.ylim(0, 1)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, value + 0.02, f'{value:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.close()

def plot_latency_over_time():
    metrics_file = 'monitoring/metrics.json'
    if not os.path.exists(metrics_file):
        return
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    if 'timestamp' not in metrics or 'prediction_latency' not in metrics:
        return
    timestamps = metrics['timestamp']
    latencies = metrics['prediction_latency']
    if len(timestamps) == 0 or len(latencies) == 0:
        return
    plt.figure(figsize=(6, 3))
    plt.plot(timestamps, latencies, marker='o', color='#4F81BD')
    plt.title('Prediction Latency Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Latency (s)')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.tight_layout()
    plt.savefig('latency_over_time.png')
    plt.close()

def main():
    plot_model_performance()
    plot_latency_over_time()
    print('Charts generated: model_performance.png, latency_over_time.png')

if __name__ == '__main__':
    main() 