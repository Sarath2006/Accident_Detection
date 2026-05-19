"""
Detailed Analysis Report Generator
Provides comprehensive detection analysis with all metrics
"""
import json
from datetime import datetime
import os


class DetailedAnalysisReport:
    """Generate detailed analysis reports with all detection metrics"""
    
    def __init__(self):
        self.analysis_data = []
    
    def add_frame_analysis(self, frame_num, metrics):
        """Add frame-level analysis"""
        self.analysis_data.append({
            'frame': frame_num,
            'timestamp': metrics.get('timestamp'),
            'cnn_score': metrics.get('cnn_score', 0.0),
            'accident_score': metrics.get('accident_score', 0.0),
            'confidence': metrics.get('confidence', 0.0),
            'speed_drops': metrics.get('speed_drops', {}),
            'collisions': metrics.get('collisions', {}),
            'trajectories': metrics.get('trajectory_anomalies', {}),
            'deformations': metrics.get('deformations', {}),
            'is_accident': metrics.get('accident', False),
            'reasons': metrics.get('reasons', []),
            'vehicles_count': metrics.get('nearby_vehicles', 0)
        })

    def _sanitize_for_json(self, value):
        if isinstance(value, dict):
            return {key: self._sanitize_for_json(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._sanitize_for_json(item) for item in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        return value
    
    def generate_summary_report(self, video_path, output_dir='analysis_reports'):
        """Generate comprehensive summary"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_name = os.path.basename(video_path)
        report_filename = f"analysis_{video_name}_{timestamp}.txt"
        report_path = os.path.join(output_dir, report_filename)
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DETAILED ACCIDENT DETECTION ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Video: {video_name}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames Analyzed: {len(self.analysis_data)}\n\n")
            
            # Statistics
            accident_frames = sum(1 for d in self.analysis_data if d['is_accident'])
            avg_cnn = sum(d['cnn_score'] for d in self.analysis_data) / len(self.analysis_data) if self.analysis_data else 0
            avg_accident_score = sum(d['accident_score'] for d in self.analysis_data) / len(self.analysis_data) if self.analysis_data else 0
            max_accident_score = max((d['accident_score'] for d in self.analysis_data), default=0)
            
            f.write("STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Accident Frames Detected: {accident_frames}\n")
            f.write(f"Average CNN Score: {avg_cnn:.3f}\n")
            f.write(f"Average Accident Score: {avg_accident_score:.2f}/10\n")
            f.write(f"Maximum Accident Score: {max_accident_score:.2f}/10\n\n")
            
            # Detailed frame analysis
            f.write("DETAILED FRAME-BY-FRAME ANALYSIS\n")
            f.write("-"*80 + "\n")
            
            for data in self.analysis_data:
                if data['cnn_score'] > 0.3 or data['accident_score'] > 2.0:  # High-interest frames
                    f.write(f"\nFrame {data['frame']}:\n")
                    f.write(f"  CNN Score: {data['cnn_score']:.3f}\n")
                    f.write(f"  Accident Score: {data['accident_score']:.2f}/10\n")
                    f.write(f"  Confidence: {data['confidence']:.1%}\n")
                    f.write(f"  Vehicles: {data['vehicles_count']}\n")
                    
                    if data['speed_drops']:
                        f.write(f"  Speed Drops: {len(data['speed_drops'])} vehicle(s)\n")
                    if data['collisions']:
                        f.write(f"  Collisions: {len(data['collisions'])} collision(s)\n")
                    if data['trajectories']:
                        f.write(f"  Trajectory Anomalies: {len(data['trajectories'])} vehicle(s)\n")
                    if data['reasons']:
                        f.write(f"  Reasons: {', '.join(data['reasons'])}\n")
                    
                    f.write(f"  Status: {'ACCIDENT' if data['is_accident'] else 'NORMAL'}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
        
        return report_path
    
    def generate_json_report(self, output_dir='analysis_reports'):
        """Export all data as JSON for further analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"analysis_data_{timestamp}.json"
        report_path = os.path.join(output_dir, report_filename)
        
        with open(report_path, 'w') as f:
            json.dump(self._sanitize_for_json(self.analysis_data), f, indent=2)
        
        return report_path
