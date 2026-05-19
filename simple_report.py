"""
Simple Report Generator for Accident Detection Results
Generates text-based accident detection reports
"""
from datetime import datetime
import os


class SimpleReportGenerator:
    """Generate text reports of accident detection results"""
    
    def __init__(self):
        self.results = []
    
    def add_video_result(
        self,
        video_name,
        category,
        accident_detected,
        severity,
        confidence,
        accident_frame_path,
        vehicles_involved,
        fire_detected=False,
        fire_confidence=0.0,
        action_required=False
    ):
        """Add a video processing result"""
        self.results.append({
            'video_name': video_name,
            'category': category,
            'accident_detected': accident_detected,
            'severity': severity,
            'confidence': confidence,
            'accident_frame_path': accident_frame_path,
            'vehicles_involved': vehicles_involved,
            'fire_detected': fire_detected,
            'fire_confidence': fire_confidence,
            'action_required': action_required,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def print_report(self):
        """Print report to console"""
        print("\n" + "="*80)
        print("ACCIDENT DETECTION REPORT")
        print("="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Videos Processed: {len(self.results)}")
        print("="*80 + "\n")
        
        accidents_count = sum(1 for r in self.results if r['accident_detected'])
        print(f"SUMMARY: {accidents_count} accident(s) detected out of {len(self.results)} video(s)\n")
        
        for i, result in enumerate(self.results, 1):
            status = "ACCIDENT DETECTED" if result['accident_detected'] else "NORMAL"
            print(f"Video {i}: {result['video_name']}")
            print(f"  Status: {status}")
            print(f"  Category: {result['category']}")
            print(f"  Severity: {result['severity']}")
            print(f"  Confidence: {result['confidence']:.1%}")
            if result.get('fire_detected'):
                print(f"  Fire Detected: YES ({result.get('fire_confidence', 0.0):.1%})")
                if result.get('action_required'):
                    print("  Action: IMMEDIATE ACTION NEEDED")
            if result['vehicles_involved']:
                print(f"  Vehicles: {', '.join(result['vehicles_involved'])}")
            if result['accident_frame_path']:
                print(f"  Frame: {result['accident_frame_path']}")
            print(f"  Time: {result['timestamp']}")
            print()
    
    def save_report(self, filename="accident_detection_report.txt"):
        """Save report to text file"""
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ACCIDENT DETECTION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Videos Processed: {len(self.results)}\n")
            f.write("="*80 + "\n\n")
            
            accidents_count = sum(1 for r in self.results if r['accident_detected'])
            f.write(f"SUMMARY: {accidents_count} accident(s) detected out of {len(self.results)} video(s)\n\n")
            
            for i, result in enumerate(self.results, 1):
                status = "ACCIDENT DETECTED" if result['accident_detected'] else "NORMAL"
                f.write(f"Video {i}: {result['video_name']}\n")
                f.write(f"  Status: {status}\n")
                f.write(f"  Category: {result['category']}\n")
                f.write(f"  Severity: {result['severity']}\n")
                f.write(f"  Confidence: {result['confidence']:.1%}\n")
                if result.get('fire_detected'):
                    f.write(f"  Fire Detected: YES ({result.get('fire_confidence', 0.0):.1%})\n")
                    if result.get('action_required'):
                        f.write("  Action: IMMEDIATE ACTION NEEDED\n")
                if result['vehicles_involved']:
                    f.write(f"  Vehicles: {', '.join(result['vehicles_involved'])}\n")
                if result['accident_frame_path']:
                    f.write(f"  Frame: {result['accident_frame_path']}\n")
                f.write(f"  Time: {result['timestamp']}\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\nReport saved to: {filename}")
        return filename
