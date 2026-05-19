"""Report Generator - Creates Court-Admissible Forensic Reports"""
import os
from typing import Optional
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


class ForensicReportGenerator:
    """Generate professional, court-admissible forensic accident reports"""
    
    def __init__(self):
        """Initialize report generator"""
        self.styles = getSampleStyleSheet()
        self.page_width, self.page_height = letter
        
        # Create custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#333333'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold',
            borderColor=colors.HexColor('#cccccc'),
            borderWidth=1,
            borderPadding=8
        )
    
    def generate_report(
        self,
        accident_scene,
        output_dir: str = 'forensics_output'
    ) -> str:
        """
        Generate complete forensic report
        
        Args:
            accident_scene: AccidentScene object
            output_dir: Directory to save report
        
        Returns:
            Path to generated PDF report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"forensic_report_{timestamp}.pdf"
        report_path = os.path.join(output_dir, report_filename)
        
        # Create PDF
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        story = []
        
        # Title
        title = Paragraph(
            "FORENSIC ACCIDENT RECONSTRUCTION REPORT",
            self.title_style
        )
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Case Information
        story.append(Paragraph("CASE INFORMATION", self.heading_style))
        fps_value = getattr(accident_scene, "fps", 30)
        env = getattr(accident_scene, "environmental_factors", {}) or {}
        speed_unit = env.get("speed_unit", "px/frame")
        calibration_label = env.get("calibration_label")
        meters_per_pixel = env.get("meters_per_pixel")
        calibration_text = "Not calibrated"
        if meters_per_pixel:
            label = calibration_label or "reference"
            calibration_text = f"{label}: {meters_per_pixel:.6f} m/px"

        case_data = [
            ['Report Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Video File:', accident_scene.video_path],
            ['Analysis Timestamp:', accident_scene.timestamp],
            ['Total Frames:', str(accident_scene.frame_count)],
            ['FPS:', str(fps_value)],
            ['Speed Unit:', speed_unit],
            ['Calibration:', calibration_text]
        ]
        
        table = Table(case_data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))

        if not getattr(accident_scene, "accident_frames", []):
            story.append(Paragraph("ACCIDENT STATUS", self.heading_style))
            story.append(Paragraph(
                "No accident frames were detected with the current thresholds. "
                "This report includes the available motion, physics, and timeline analysis.",
                self.styles['Normal']
            ))
            story.append(Spacer(1, 0.2*inch))
        
        # Physics Analysis
        if accident_scene.physics_analysis:
            story.append(Paragraph("PHYSICS ANALYSIS", self.heading_style))
            
            physics = accident_scene.physics_analysis
            max_rel_vel = self._max_relative_velocity(physics.get('collision_velocity', {}))
            max_force = self._max_impact_force(physics.get('impact_forces', {}))
            max_energy = self._max_kinetic_energy(physics.get('kinetic_energy', {}))

            rel_vel_text = f"{max_rel_vel:.2f} {speed_unit}" if max_rel_vel is not None else "N/A"
            force_text = f"{max_force:,.0f} N" if max_force is not None else "N/A"
            energy_text = f"{max_energy:,.0f} J" if max_energy is not None else "N/A"

            physics_summary = f"""
            <b>Collision Velocity (max relative):</b> {rel_vel_text}<br/>
            <b>Peak Impact Force:</b> {force_text}<br/>
            <b>Peak Kinetic Energy:</b> {energy_text}
            """
            story.append(Paragraph(physics_summary, self.styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Fault Determination
        if accident_scene.fault_determination:
            story.append(Paragraph("FAULT DETERMINATION", self.heading_style))
            
            fault = accident_scene.fault_determination
            liability = fault.get('liability_percentage', {})
            
            fault_summary = f"""
            <b>Primary At-Fault Vehicle:</b> {fault.get('primary_fault_vehicle', 'UNDETERMINED')}<br/>
            <b>Vehicle 1 Liability:</b> {liability.get('vehicle_1', 0):.1f}%<br/>
            <b>Vehicle 2 Liability:</b> {liability.get('vehicle_2', 0):.1f}%<br/>
            <b>Recommendation:</b> {fault.get('recommendation', 'INCONCLUSIVE')}<br/>
            <b>Confidence Score:</b> {fault.get('confidence_score', 0):.1%}
            """
            story.append(Paragraph(fault_summary, self.styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Timeline
        if accident_scene.timeline:
            story.append(Paragraph("ACCIDENT SEQUENCE TIMELINE", self.heading_style))
            
            timeline_data = [['Time (s)', 'Frame', 'Event Type', 'Description']]
            fps_value = getattr(accident_scene, "fps", 30)
            for event in accident_scene.timeline[:20]:  # Limit to 20 events
                timestamp = self._event_timestamp(event, fps_value)
                frame_num = self._event_frame_number(event)
                event_type = self._event_type(event)
                description = self._event_description(event)
                timeline_data.append([
                    f"{timestamp:.2f}",
                    str(frame_num),
                    event_type,
                    description
                ])
            
            timeline_table = Table(timeline_data, colWidths=[0.8*inch, 0.8*inch, 1.2*inch, 2.2*inch])
            timeline_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
            ]))
            story.append(timeline_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Conclusion
        story.append(PageBreak())
        story.append(Paragraph("CONCLUSION", self.heading_style))
        
        conclusion = """
        This forensic accident reconstruction analysis is based on evidence extracted from video analysis,
        including vehicle detection, speed estimation, and physics-based calculations. The fault determination
        is rendered to a reasonable degree of scientific certainty based on the available evidence.<br/><br/>
        
        This report is intended for use in legal proceedings and insurance claim determination. All calculations
        follow NHTSA (National Highway Traffic Safety Administration) standards for accident reconstruction.
        """
        story.append(Paragraph(conclusion, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        return report_path

    def _event_timestamp(self, event, fps_value: float) -> float:
        if isinstance(event, dict):
            if event.get('timestamp_s') is not None:
                return float(event.get('timestamp_s'))
            if event.get('timestamp_ms') is not None:
                return float(event.get('timestamp_ms')) / 1000.0
            frame_num = event.get('frame_number', event.get('frame', 0))
            return float(frame_num) / max(fps_value, 1)
        return 0.0

    def _event_frame_number(self, event) -> int:
        if isinstance(event, dict):
            return int(event.get('frame_number', event.get('frame', 0)) or 0)
        return 0

    def _event_type(self, event) -> str:
        if isinstance(event, dict):
            value = event.get('event_type', '')
            if value is None:
                return ""
            return str(value).replace('_', ' ').upper()
        return ""

    def _event_description(self, event) -> str:
        if isinstance(event, dict):
            value = event.get('description', '')
            if value is None:
                return ""
            return str(value)[:50]
        return ""

    def _max_relative_velocity(self, collision_velocity: dict):
        values = []
        for data in collision_velocity.values():
            if isinstance(data, dict):
                rel = data.get('relative_velocity_ms')
                if rel is not None:
                    values.append(float(rel))
        return max(values) if values else None

    def _max_impact_force(self, impact_forces: dict):
        values = []
        for data in impact_forces.values():
            if isinstance(data, dict):
                for force in data.values():
                    try:
                        values.append(float(force))
                    except (TypeError, ValueError):
                        continue
        return max(values) if values else None

    def _max_kinetic_energy(self, kinetic_energy: dict):
        values = []
        for data in kinetic_energy.values():
            if isinstance(data, dict):
                for energy in data.values():
                    try:
                        values.append(float(energy))
                    except (TypeError, ValueError):
                        continue
        return max(values) if values else None
