"""
Generate Word Document Report for HMM Activity Recognition Project.
"""

from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR


def create_report():
    """Generate the HMM Activity Recognition report as a Word document."""
    
    doc = Document()
    
    # Set up styles
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(12)
    
    # ==================== TITLE PAGE ====================
    doc.add_paragraph()
    doc.add_paragraph()
    
    title = doc.add_paragraph()
    title_run = title.add_run("Human Activity Recognition Using Hidden Markov Models")
    title_run.bold = True
    title_run.font.size = Pt(24)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    subtitle = doc.add_paragraph()
    subtitle_run = subtitle.add_run("Smartphone Sensor-Based Activity Classification")
    subtitle_run.font.size = Pt(16)
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph()
    doc.add_paragraph()
    
    # Group info
    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.add_run("Group Members:\n").bold = True
    info.add_run("Mathias\n\n")
    info.add_run("Course: Machine Learning / Data Science\n")
    info.add_run("Date: March 2026\n")
    
    doc.add_page_break()
    
    # ==================== TABLE OF CONTENTS ====================
    toc_heading = doc.add_heading("Table of Contents", level=1)
    
    toc_items = [
        ("1. Background and Motivation", 3),
        ("2. Data Collection and Preprocessing", 3),
        ("   2.1 Sensor Configuration", 4),
        ("   2.2 Data Format", 4),
        ("   2.3 Preprocessing Pipeline", 4),
        ("3. HMM Setup and Implementation", 5),
        ("   3.1 Model Components", 5),
        ("   3.2 Algorithm Implementation", 6),
        ("4. Results and Interpretation", 7),
        ("   4.1 Evaluation Metrics", 7),
        ("   4.2 Transition Matrix Analysis", 8),
        ("   4.3 Confusion Matrix", 8),
        ("5. Discussion and Conclusion", 9),
    ]
    
    for item, page in toc_items:
        p = doc.add_paragraph()
        p.add_run(f"{item}")
        tab_stops = p.paragraph_format.tab_stops
        p.add_run(f"\t{page}")
    
    doc.add_page_break()
    
    # ==================== 1. BACKGROUND AND MOTIVATION ====================
    doc.add_heading("1. Background and Motivation", level=1)
    
    background_text = """Human Activity Recognition (HAR) has become increasingly important in modern applications, ranging from healthcare monitoring and fitness tracking to smart home automation and elderly care systems. Our group's unique use case focuses on developing a smartphone-based activity recognition system that can accurately classify four fundamental human activities: standing, walking, jumping, and remaining still. This use case is particularly relevant because smartphones are ubiquitous devices equipped with inertial measurement units (IMUs) including accelerometers and gyroscopes, making activity recognition accessible without requiring specialized wearable hardware. The ability to automatically recognize these activities enables numerous practical applications: fitness applications can automatically track workout sessions and count exercises; healthcare systems can monitor patient mobility and detect falls; smart home systems can adjust lighting and climate based on occupant activities; and research studies can collect objective activity data without manual logging. By implementing a Hidden Markov Model (HMM) approach, we aim to leverage the temporal dependencies inherent in human activities, as activities are not instantaneous events but rather sequences of movements that evolve over time. The HMM framework naturally captures these temporal patterns through its state transition probabilities, making it well-suited for sequential activity classification from continuous sensor streams."""
    
    p = doc.add_paragraph(background_text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    # ==================== 2. DATA COLLECTION AND PREPROCESSING ====================
    doc.add_heading("2. Data Collection and Preprocessing", level=1)
    
    doc.add_heading("2.1 Sensor Configuration", level=2)
    
    sensor_intro = """Data was collected using smartphone sensors with the following configuration:"""
    doc.add_paragraph(sensor_intro)
    
    # Sensor configuration table
    table = doc.add_table(rows=5, cols=2)
    table.style = 'Table Grid'
    
    sensor_data = [
        ("Parameter", "Value"),
        ("Phone Model", "OPPO CPH2641"),
        ("Sampling Rate", "100 Hz (10ms intervals)"),
        ("Sensors Used", "Accelerometer (x, y, z), Gyroscope (x, y, z)"),
        ("Recording Duration", "~20 seconds per activity"),
    ]
    
    for i, (param, value) in enumerate(sensor_data):
        row = table.rows[i]
        row.cells[0].text = param
        row.cells[1].text = value
        if i == 0:
            row.cells[0].paragraphs[0].runs[0].bold = True
            row.cells[1].paragraphs[0].runs[0].bold = True
    
    doc.add_paragraph()
    
    # Activities table
    doc.add_paragraph("The following activities were recorded:")
    
    table2 = doc.add_table(rows=5, cols=3)
    table2.style = 'Table Grid'
    
    activity_data = [
        ("Activity", "Duration", "Notes"),
        ("Standing", "5-10 seconds", "Phone steady at waist level"),
        ("Walking", "5-10 seconds", "Consistent walking pace"),
        ("Jumping", "5-10 seconds", "Continuous vertical jumps"),
        ("Still (No Movement)", "5-10 seconds", "Phone on flat surface"),
    ]
    
    for i, row_data in enumerate(activity_data):
        row = table2.rows[i]
        for j, cell_text in enumerate(row_data):
            row.cells[j].text = cell_text
            if i == 0:
                row.cells[j].paragraphs[0].runs[0].bold = True
    
    doc.add_paragraph()
    
    doc.add_heading("2.2 Data Format", level=2)
    
    data_format_text = """Each recording session generated CSV files containing timestamped sensor readings. The accelerometer data captures linear acceleration along three orthogonal axes (x, y, z) measured in m/s², while the gyroscope measures angular velocity in rad/s. The data files follow a consistent naming convention: ParticipantName_activity-YYYY-MM-DD_HH-MM-SS, enabling easy identification and organization of recordings."""
    doc.add_paragraph(data_format_text)
    
    doc.add_paragraph("Sample data structure:")
    
    # Code block for data format
    code_text = """Accelerometer.csv columns: time, seconds_elapsed, z, y, x
Gyroscope.csv columns: time, seconds_elapsed, z, y, x
Metadata.csv: device info, sampling rate, sensors used"""
    
    code_para = doc.add_paragraph()
    code_run = code_para.add_run(code_text)
    code_run.font.name = 'Courier New'
    code_run.font.size = Pt(10)
    
    doc.add_paragraph()
    
    doc.add_heading("2.3 Preprocessing Pipeline", level=2)
    
    preprocessing_text = """The preprocessing pipeline consists of several stages to prepare raw sensor data for feature extraction:"""
    doc.add_paragraph(preprocessing_text)
    
    preprocessing_steps = [
        "Data Loading and Merging: Accelerometer and gyroscope data are loaded and merged based on timestamps using nearest-neighbor matching to handle slight timing differences between sensors.",
        "Low-pass Filtering: A 4th-order Butterworth low-pass filter with a 20 Hz cutoff frequency is applied to reduce high-frequency noise while preserving activity-related signal components.",
        "Windowing: The continuous sensor stream is segmented into overlapping windows of 500ms (50 samples at 100 Hz) with 50% overlap (250ms). This window size balances temporal resolution with capturing sufficient activity patterns.",
        "Feature Extraction: From each window, 130 features are computed combining time-domain and frequency-domain characteristics.",
        "Normalization: Z-score normalization is applied to ensure features have zero mean and unit variance, preventing features with larger magnitudes from dominating the model."
    ]
    
    for i, step in enumerate(preprocessing_steps, 1):
        p = doc.add_paragraph(style='List Number')
        p.add_run(step)
    
    doc.add_page_break()
    
    # ==================== 3. HMM SETUP AND IMPLEMENTATION ====================
    doc.add_heading("3. HMM Setup and Implementation", level=1)
    
    doc.add_heading("3.1 Model Components", level=2)
    
    hmm_intro = """The Hidden Markov Model provides a probabilistic framework for modeling sequential data where the underlying states are not directly observable. Our HMM is defined by the following components:"""
    doc.add_paragraph(hmm_intro)
    
    # HMM components table
    table3 = doc.add_table(rows=6, cols=2)
    table3.style = 'Table Grid'
    
    hmm_components = [
        ("Component", "Description"),
        ("Hidden States (Z)", "The four activities: standing, walking, jumping, still. These represent the true activity being performed but are not directly observed."),
        ("Observations (X)", "130-dimensional feature vectors extracted from each window of accelerometer and gyroscope data."),
        ("Transition Probabilities (A)", "A 4×4 matrix where A[i,j] represents P(state_t = j | state_{t-1} = i), the probability of transitioning from activity i to activity j."),
        ("Emission Probabilities (B)", "Gaussian distributions modeling P(observation | state). Each state has a mean vector and diagonal covariance matrix."),
        ("Initial Probabilities (π)", "P(state_0), the probability of starting in each activity state. Initialized uniformly as 0.25 for each state."),
    ]
    
    for i, (component, desc) in enumerate(hmm_components):
        row = table3.rows[i]
        row.cells[0].text = component
        row.cells[1].text = desc
        if i == 0:
            row.cells[0].paragraphs[0].runs[0].bold = True
            row.cells[1].paragraphs[0].runs[0].bold = True
    
    doc.add_paragraph()
    
    # Feature categories
    doc.add_paragraph("Features extracted from each window include:")
    
    doc.add_paragraph("Time-Domain Features:", style='List Bullet')
    time_features = doc.add_paragraph("Mean, standard deviation, variance, min, max, range for each axis; Signal Magnitude Area (SMA); correlation between axes (xy, xz, yz); zero-crossing rate; RMS; skewness; kurtosis; magnitude statistics.")
    
    doc.add_paragraph("Frequency-Domain Features:", style='List Bullet')
    freq_features = doc.add_paragraph("Dominant frequency; spectral energy; first 5 FFT coefficients; spectral entropy; mean frequency; energy in frequency bands (0-2 Hz, 2-5 Hz, 5-15 Hz).")
    
    doc.add_heading("3.2 Algorithm Implementation", level=2)
    
    algo_intro = """Two key algorithms were implemented for training and inference:"""
    doc.add_paragraph(algo_intro)
    
    doc.add_paragraph().add_run("Baum-Welch Algorithm (Training):").bold = True
    baum_welch = """The Baum-Welch algorithm is an Expectation-Maximization (EM) procedure used to estimate the HMM parameters from training data. In the E-step, we compute the forward probabilities α(t,i) = P(o_1,...,o_t, s_t=i) and backward probabilities β(t,i) = P(o_{t+1},...,o_T | s_t=i) using dynamic programming. These are combined to compute the state posteriors γ(t,i) = P(s_t=i | O) and transition posteriors ξ(t,i,j) = P(s_t=i, s_{t+1}=j | O). In the M-step, we update the model parameters: transition probabilities are updated as the expected number of transitions from state i to j divided by the expected number of transitions from state i; emission parameters (means and covariances) are updated using the weighted observations."""
    doc.add_paragraph(baum_welch)
    
    doc.add_paragraph().add_run("Viterbi Algorithm (Decoding):").bold = True
    viterbi = """The Viterbi algorithm finds the most likely sequence of hidden states given the observations. It uses dynamic programming where V(t,j) represents the probability of the most likely path ending in state j at time t. The algorithm maintains backpointers to reconstruct the optimal path. The complexity is O(T × N²) where T is the sequence length and N is the number of states."""
    doc.add_paragraph(viterbi)
    
    impl_details = """The implementation uses log-probabilities throughout to prevent numerical underflow with long sequences. The model converged after 19 iterations of Baum-Welch training with a convergence tolerance of 1e-4."""
    doc.add_paragraph(impl_details)
    
    doc.add_page_break()
    
    # ==================== 4. RESULTS AND INTERPRETATION ====================
    doc.add_heading("4. Results and Interpretation", level=1)
    
    doc.add_heading("4.1 Evaluation Metrics", level=2)
    
    results_intro = """The model was trained on 268 windows (80% of data) and evaluated on 67 windows (20% of data). The following table summarizes the per-activity performance:"""
    doc.add_paragraph(results_intro)
    
    # Results table
    table4 = doc.add_table(rows=6, cols=5)
    table4.style = 'Table Grid'
    
    results_data = [
        ("Activity", "Samples", "Sensitivity", "Specificity", "F1-Score"),
        ("Standing", "17", "100.00%", "64.00%", "0.65"),
        ("Walking", "17", "0.00%", "92.00%", "0.00"),
        ("Jumping", "17", "70.59%", "100.00%", "0.83"),
        ("Still", "16", "100.00%", "100.00%", "1.00"),
        ("Overall", "67", "-", "-", "67.16%"),
    ]
    
    for i, row_data in enumerate(results_data):
        row = table4.rows[i]
        for j, cell_text in enumerate(row_data):
            row.cells[j].text = cell_text
            if i == 0 or i == 5:
                row.cells[j].paragraphs[0].runs[0].bold = True
    
    doc.add_paragraph()
    
    results_analysis = """Key observations from the results:

The model achieves perfect recognition (100% sensitivity) for "Still" and "Standing" activities. The "Still" activity is particularly easy to classify due to its distinctive near-zero variance pattern when the phone is stationary.

"Jumping" shows good performance with 70.59% sensitivity and perfect specificity, correctly identifying most jumping instances while never misclassifying other activities as jumping.

"Walking" presents the greatest challenge with 0% sensitivity, indicating all walking samples were misclassified. Analysis of the confusion matrix reveals that walking is being confused with standing, likely due to the similar phone orientation and the limited number of training samples (only one recording per activity)."""
    doc.add_paragraph(results_analysis)
    
    doc.add_heading("4.2 Transition Matrix Analysis", level=2)
    
    transition_text = """The learned transition matrix reveals the probability of transitioning between activities:"""
    doc.add_paragraph(transition_text)
    
    # Transition matrix table
    table5 = doc.add_table(rows=5, cols=5)
    table5.style = 'Table Grid'
    
    trans_data = [
        ("From \\ To", "Standing", "Walking", "Jumping", "Still"),
        ("Standing", "0.46", "0.22", "0.25", "0.07"),
        ("Walking", "0.50", "0.19", "0.25", "0.06"),
        ("Jumping", "0.55", "0.14", "0.23", "0.09"),
        ("Still", "0.63", "0.05", "0.26", "0.05"),
    ]
    
    for i, row_data in enumerate(trans_data):
        row = table5.rows[i]
        for j, cell_text in enumerate(row_data):
            row.cells[j].text = cell_text
            if i == 0 or j == 0:
                row.cells[j].paragraphs[0].runs[0].bold = True
    
    doc.add_paragraph()
    
    trans_analysis = """The transition matrix shows moderate self-transition probabilities (diagonal elements ranging from 0.19 to 0.46), indicating the model has learned some temporal persistence in activities. The relatively lower self-transition probabilities compared to ideal scenarios (typically >0.8) suggest the model would benefit from more sequential training data to better capture activity persistence."""
    doc.add_paragraph(trans_analysis)
    
    doc.add_heading("4.3 Confusion Matrix", level=2)
    
    confusion_text = """The confusion matrix provides detailed insight into classification errors:

- Standing: All 17 samples correctly classified as standing
- Walking: All 17 samples misclassified (16 as standing, 1 as jumping)  
- Jumping: 12 correctly classified, 5 misclassified as standing
- Still: All 16 samples correctly classified

The primary source of confusion is between walking and standing, which can be attributed to similar phone orientations during both activities when the phone is held at waist level."""
    doc.add_paragraph(confusion_text)
    
    doc.add_page_break()
    
    # ==================== 5. DISCUSSION AND CONCLUSION ====================
    doc.add_heading("5. Discussion and Conclusion", level=1)
    
    doc.add_paragraph().add_run("Activity Distinguishability:").bold = True
    distinguish_text = """The "Still" activity proved easiest to distinguish due to its minimal sensor variance - when the phone is stationary on a flat surface, the accelerometer readings show only gravitational acceleration with negligible variation. "Jumping" was relatively distinguishable due to its characteristic high-amplitude vertical acceleration spikes during takeoff and landing phases.

The most challenging distinction was between standing and walking. Both activities involve the phone being held upright at waist level with similar orientations. Walking introduces periodic oscillations in acceleration (~2 Hz corresponding to step frequency), but with limited training data, the model struggled to capture these subtle distinctions."""
    doc.add_paragraph(distinguish_text)
    
    doc.add_paragraph().add_run("Impact of Sensor Noise and Sampling:").bold = True
    noise_text = """Several factors affected model performance:

1. Sensor Noise: Smartphone IMU sensors exhibit inherent noise and drift, particularly in the gyroscope. Low-pass filtering helped mitigate high-frequency noise but may have attenuated rapid movement signatures.

2. Phone Placement Variability: The exact position and orientation of the phone affects sensor readings. Standardized placement protocols would improve consistency.

3. Sampling Rate Harmonization: When collecting data from multiple phones with different native sampling rates, resampling to a common rate (100 Hz) is essential for consistent feature extraction.

4. Limited Data: With only one recording per activity (~20 seconds each), the model had limited examples to learn the full variability of each activity pattern."""
    doc.add_paragraph(noise_text)
    
    doc.add_paragraph().add_run("Recommendations for Improvement:").bold = True
    
    improvements = [
        "Collect More Data: Increase to 10-15 recordings per activity from multiple participants to capture greater variability in movement patterns.",
        "Add Participants: Include recordings from different group members with varying heights, walking speeds, and movement styles.",
        "Extended Duration: Longer recordings (30+ seconds) would provide more windows for training and better capture activity transitions.",
        "Additional Features: Consider adding wavelet transforms, autocorrelation features, or statistical moments of higher order.",
        "Hierarchical HMM: Implement a two-level HMM where low-level states capture sub-movements and high-level states represent activities.",
        "Deep Learning Hybrid: Combine CNN/LSTM feature extractors with HMM temporal modeling for potentially better performance."
    ]
    
    for imp in improvements:
        doc.add_paragraph(imp, style='List Bullet')
    
    doc.add_paragraph().add_run("Conclusion:").bold = True
    conclusion = """This project successfully implemented a Hidden Markov Model for smartphone-based activity recognition. The system demonstrates the viability of using probabilistic graphical models for classifying human activities from inertial sensor data. While the current model achieved 67.16% overall accuracy with limited training data, it showed perfect recognition for stationary activities (still, standing) and good performance for jumping. The primary limitation—confusion between walking and standing—can be addressed through expanded data collection. The implemented Viterbi and Baum-Welch algorithms provide a solid foundation for temporal sequence modeling, and the feature extraction pipeline captures both time and frequency domain characteristics essential for activity discrimination. Future work should focus on data augmentation, cross-participant validation, and exploration of hybrid deep learning approaches."""
    doc.add_paragraph(conclusion)
    
    # ==================== REFERENCES ====================
    doc.add_page_break()
    doc.add_heading("References", level=1)
    
    references = [
        "Rabiner, L. R. (1989). A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition. Proceedings of the IEEE, 77(2), 257-286.",
        "Lara, O. D., & Labrador, M. A. (2013). A Survey on Human Activity Recognition using Wearable Sensors. IEEE Communications Surveys & Tutorials, 15(3), 1192-1209.",
        "Bulling, A., Blanke, U., & Schiele, B. (2014). A Tutorial on Human Activity Recognition Using Body-worn Inertial Sensors. ACM Computing Surveys, 46(3), 1-33.",
        "hmmlearn Documentation. https://hmmlearn.readthedocs.io/",
    ]
    
    for i, ref in enumerate(references, 1):
        doc.add_paragraph(f"[{i}] {ref}")
    
    # ==================== APPENDIX ====================
    doc.add_heading("Appendix: Data Collection Details", level=1)
    
    appendix_table = doc.add_table(rows=3, cols=4)
    appendix_table.style = 'Table Grid'
    
    appendix_data = [
        ("Group Member", "Phone Model", "Sampling Rate", "Recordings"),
        ("Mathias", "OPPO CPH2641", "100 Hz (10ms)", "4 activities"),
        ("[Partner Name]", "[Phone Model]", "[Rate]", "[Activities]"),
    ]
    
    for i, row_data in enumerate(appendix_data):
        row = appendix_table.rows[i]
        for j, cell_text in enumerate(row_data):
            row.cells[j].text = cell_text
            if i == 0:
                row.cells[j].paragraphs[0].runs[0].bold = True
    
    # Save document
    output_path = os.path.join(RESULTS_DIR, "HMM_Activity_Recognition_Report.docx")
    doc.save(output_path)
    print(f"\nReport saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    create_report()
