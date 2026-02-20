import sqlite3
import os
from datetime import datetime
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(BASE_DIR, "logs")
db_path = os.path.join(logs_dir, "predictions.db")

def init_db():
    """Initialize database with predictions table"""
    # Create logs directory if it doesn't exist
    os.makedirs(logs_dir, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create predictions table
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp TEXT,
            ip_address TEXT,
            user_agent TEXT
        )
    """)
    
    # Create index on timestamp for faster queries
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON predictions(timestamp)
    """)
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database initialized at {db_path}")

def log_prediction(text, prediction, confidence, ip_address=None, user_agent=None):
    """
    Log a prediction to the database
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute("""
        INSERT INTO predictions (text, prediction, confidence, timestamp, ip_address, user_agent)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        text[:500],  # Limit text length
        prediction, 
        confidence, 
        datetime.now().isoformat(),
        ip_address,
        user_agent
    ))
    
    conn.commit()
    conn.close()

def get_recent_predictions(limit=10):
    """
    Get recent predictions from database
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("""
        SELECT id, text, prediction, confidence, timestamp 
        FROM predictions 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (limit,))
    
    rows = c.fetchall()
    conn.close()
    
    # Convert to list of dictionaries
    result = []
    for row in rows:
        result.append({
            'id': row['id'],
            'text': row['text'][:100] + '...' if len(row['text']) > 100 else row['text'],
            'prediction': row['prediction'],
            'confidence': row['confidence'],
            'timestamp': row['timestamp']
        })
    
    return result

def get_prediction_count():
    """Get total number of predictions"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM predictions")
    count = c.fetchone()[0]
    
    conn.close()
    return count

def get_statistics():
    """Get prediction statistics"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Total predictions
    c.execute("SELECT COUNT(*) FROM predictions")
    total = c.fetchone()[0]
    
    if total == 0:
        conn.close()
        return {
            'total_predictions': 0,
            'by_class': {},
            'average_confidence': 0,
            'daily_predictions': {}
        }
    
    # Predictions by class
    c.execute("""
        SELECT prediction, COUNT(*) as count 
        FROM predictions 
        GROUP BY prediction
    """)
    by_class = dict(c.fetchall())
    
    # Average confidence
    c.execute("SELECT AVG(confidence) FROM predictions")
    avg_confidence = c.fetchone()[0] or 0
    
    # Predictions by day (last 7 days)
    c.execute("""
        SELECT DATE(timestamp) as day, COUNT(*) 
        FROM predictions 
        WHERE timestamp >= DATE('now', '-7 days')
        GROUP BY DATE(timestamp)
        ORDER BY day DESC
    """)
    daily = dict(c.fetchall())
    
    conn.close()
    
    return {
        'total_predictions': total,
        'by_class': by_class,
        'average_confidence': round(avg_confidence, 3),
        'daily_predictions': daily
    }