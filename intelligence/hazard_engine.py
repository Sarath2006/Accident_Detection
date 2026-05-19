class HazardEngine:
    """
    Final decision-making engine for hazard detection.
    
    Severity Classification (Real-World):
    - MINOR: Small collision, minor damage, no rollover
    - MODERATE: Medium impact, vehicle damage, potential injuries
    - SEVERE: High-speed collision, rollover, fire, multiple vehicles
    """

    def __init__(self):
        self.previous_state = "NORMAL"

    def decide(self, accident_event):
        """
        Produces final hazard decision with severity and confidence.
        Uses accident_score and severity_indicators from accident logic.
        """
        if accident_event["accident"]:
            severity, confidence = self._estimate_severity(accident_event)

            speed_drops = accident_event.get("speed_drops") or accident_event.get("sudden_stops") or {}
            trajectory_breaks = accident_event.get("trajectory_anomalies") or accident_event.get("trajectory_breaks") or {}

            decision = {
                "status": "ACCIDENT",
                "severity": severity,
                "confidence": confidence,
                "reasons": accident_event["reasons"],
                "accident_score": accident_event.get("accident_score", 0),
                "cnn_score": accident_event.get("cnn_score", 0.0),
                "collisions": len(accident_event.get("collisions", [])),
                "stopped_vehicles": len(speed_drops),
                "trajectory_breaks": len(trajectory_breaks)
            }

            self.previous_state = "ACCIDENT"
            return decision

        self.previous_state = "NORMAL"
        return {"status": "NORMAL"}

    def _estimate_severity(self, event):
        """
        Severity estimation based on real-world damage indicators.
        
        SEVERE (Rollover, high-speed, fire, multiple damage):
        - Fire/smoke present
        - accident_score ≥ 8
        - High collision overlap (>0.5 IoU) + speed drop
        - Multiple collisions
        - Trajectory break + collision (spin/rollover)
        
        MODERATE (Medium damage, vehicle damaged):
        - Collision detected + speed drop
        - High speed drop severity
        - Multiple vehicles affected
        - accident_score 6-7
        
        MINOR (Small collision, minimal damage):
        - Low collision overlap (<0.4)
        - Slow speed collision
        - Single vehicle, minor stop
        - accident_score 5
        """
        score = event.get("accident_score", 0)
        num_collisions = len(event.get("collisions", []))
        num_stops = len(event.get("speed_drops") or event.get("sudden_stops") or {})
        trajectory_breaks = len(event.get("trajectory_anomalies") or event.get("trajectory_breaks") or {})
        reasons = event.get("reasons", [])
        has_fire_smoke = event.get("fire_or_smoke", False) or any(
            "fire" in str(reason).lower() and "smoke" in str(reason).lower()
            for reason in reasons
        )
        severity_indicators = event.get("severity_indicators", [])
        
        # Count severity indicators
        severe_count = severity_indicators.count("severe")
        moderate_count = severity_indicators.count("moderate")
        
        # SEVERE: High-impact, rollover, fire, heavy damage
        if has_fire_smoke:
            return ("SEVERE", 0.95)  # Fire = extremely severe
        
        if score >= 8:
            return ("SEVERE", 0.92)  # Very high score = severe accident
        
        # Collision + trajectory break = rollover/spin (SEVERE)
        if num_collisions >= 1 and trajectory_breaks >= 1:
            return ("SEVERE", 0.90)
        
        # Multiple collisions = severe multi-vehicle crash
        if num_collisions >= 2:
            return ("SEVERE", 0.88)
        
        # High collision severity indicators
        if severe_count >= 2:
            return ("SEVERE", 0.85)
        
        # MODERATE: Medium damage, vehicle affected
        if num_collisions >= 1 and num_stops >= 1:
            return ("MODERATE", 0.75)  # Collision causing stop
        
        if score >= 6:
            return ("MODERATE", 0.72)  # Good evidence score
        
        if moderate_count >= 2 or severe_count >= 1:
            return ("MODERATE", 0.70)
        
        # Trajectory break alone (swerve, lane departure)
        if trajectory_breaks >= 1:
            return ("MODERATE", 0.68)
        
        # Multiple vehicles involved
        if num_stops >= 3:
            return ("MODERATE", 0.65)
        
        # MINOR: Small collision, minimal damage
        if num_collisions >= 1:
            return ("MINOR", 0.60)  # Low overlap collision
        
        if num_stops >= 1:
            return ("MINOR", 0.55)  # Single vehicle stopped
        
        # Default minimal detection
        return ("MINOR", 0.50)
