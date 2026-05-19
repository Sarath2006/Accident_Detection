import sqlite3
import os
from datetime import datetime


class DatabaseManager:
    """
    Handles all database operations
    """

    def __init__(self, db_path="database/accident_detection.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        # Video-level summary (main reporting table)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS video_accidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT UNIQUE,
            category TEXT,
            accident_detected INTEGER,
            severity TEXT,
            confidence REAL,
            fire_smoke INTEGER,
            vehicles_involved TEXT,
            estimated_deaths INTEGER,
            accident_frame_path TEXT,
            timestamp TEXT
        )
        """)
        
        # Frame-level details (for debugging)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS accident_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT,
            category TEXT,
            frame_number INTEGER,
            accident_detected INTEGER,
            severity TEXT,
            cnn_score REAL,
            confidence REAL,
            fire_smoke INTEGER,
            vehicle_id TEXT,
            vehicle_class TEXT,
            vehicle_speed REAL,
            timestamp TEXT
        )
        """)
        self.conn.commit()

    def insert_video_accident(
        self,
        video_name,
        category,
        accident_detected,
        severity,
        confidence,
        fire_smoke,
        vehicles_involved,
        estimated_deaths,
        accident_frame_path
    ):
        """
        Insert or update video-level accident summary
        """
        try:
            # vehicles_involved is already a formatted string
            vehicles_str = vehicles_involved if isinstance(vehicles_involved, str) else str(vehicles_involved)
            
            self.cursor.execute("""
            INSERT OR REPLACE INTO video_accidents (
                video_name,
                category,
                accident_detected,
                severity,
                confidence,
                fire_smoke,
                vehicles_involved,
                estimated_deaths,
                accident_frame_path,
                timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video_name,
                category,
                accident_detected,
                severity,
                confidence,
                fire_smoke,
                vehicles_str,
                estimated_deaths,
                accident_frame_path,
                __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            self.conn.commit()
        except Exception as e:
            print(f"[ERROR] Failed to insert video accident: {e}")

    def insert_frame_event(
        self,
        video_name,
        category,
        frame_number,
        accident_detected,
        severity,
        cnn_score,
        confidence,
        fire_smoke,
        vehicle_id=None,
        vehicle_class=None,
        vehicle_speed=0.0
    ):
        """
        Insert frame-level event (for detailed logging)
        """
        self.cursor.execute("""
        INSERT INTO accident_events (
            video_name,
            category,
            frame_number,
            accident_detected,
            severity,
            cnn_score,
            confidence,
            fire_smoke,
            vehicle_id,
            vehicle_class,
            vehicle_speed,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            video_name,
            category,
            frame_number,
            accident_detected,
            severity,
            cnn_score,
            confidence,
            fire_smoke,
            str(vehicle_id) if vehicle_id else None,
            vehicle_class,
            vehicle_speed,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        self.conn.commit()

    def insert_event(
        self,
        video_name,
        category,
        accident_detected,
        severity,
        cnn_score,
        fire_smoke
    ):
        """Legacy method for backward compatibility"""
        self.cursor.execute("""
        INSERT INTO accident_events (
            video_name,
            category,
            accident_detected,
            severity,
            cnn_score,
            fire_smoke,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            video_name,
            category,
            accident_detected,
            severity,
            cnn_score,
            fire_smoke,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        self.conn.commit()

    def close(self):
        self.conn.close()

    def get_video_accidents(self):
        """
        Get all video-level accident summaries (MAIN REPORT)
        """
        self.cursor.execute("""
        SELECT 
            video_name, 
            category, 
            accident_detected, 
            severity, 
            confidence, 
            fire_smoke, 
            vehicles_involved, 
            estimated_deaths, 
            accident_frame_path 
        FROM video_accidents 
        ORDER BY severity DESC, confidence DESC
        """)
        return self.cursor.fetchall()

    def get_statistics(self):
        """
        Returns summary statistics from video_accidents table
        """
        self.cursor.execute("""
        SELECT 
            COUNT(*) as total_videos,
            SUM(accident_detected) as accident_videos,
            AVG(confidence) as avg_confidence,
            SUM(fire_smoke) as fire_smoke_count,
            SUM(estimated_deaths) as total_deaths
        FROM video_accidents
        """)
        
        result = self.cursor.fetchone()
        return {
            "total_videos": result[0] or 0,
            "accident_videos": result[1] or 0,
            "avg_confidence": round(result[2], 3) if result[2] else 0.0,
            "fire_smoke_count": result[3] or 0,
            "total_deaths": result[4] or 0
        }

    def get_severity_breakdown(self):
        """
        Returns accident count by severity from video-level data
        """
        self.cursor.execute("""
        SELECT severity, COUNT(*) 
        FROM video_accidents 
        WHERE accident_detected = 1
        GROUP BY severity
        """)
        
        return dict(self.cursor.fetchall())
