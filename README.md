🛡️ LayerGuard Enhanced - Advanced LLM Poison Detection Tool 

   
   

    

LayerGuard Enhanced is a cutting-edge forensic tool designed to detect poisoned weight layers in Large Language Models (LLMs). 

It provides comprehensive analysis using statistical anomaly detection, pattern recognition, and advanced entropy calculations to identify potential security threats in AI models. 
🌟 Key Features 

    Multi-Format Support: PyTorch, HuggingFace Transformers, TensorFlow checkpoints
    Advanced Anomaly Detection: Statistical analysis with Isolation Forest and Z-score methods
    Enhanced Entropy Analysis: Differential entropy for continuous weight distributions
    Layer Filtering: Regex-based layer selection for targeted analysis
    Multiple Report Formats: JSON, YAML, Markdown for different use cases
    Professional Visualization: PNG and PDF exports for presentations
    Batch Processing: Analyze multiple models simultaneously
    CI/CD Integration: Automated security pipelines for model deployment
     

📋 Table of Contents 

    Installation 
    Quick Start 
    Advanced Usage 
    Command Line Options 
    Batch Processing 
    Report Formats 
    Visualization Features 
    Technical Architecture 
    Detection Methods 
    Contributing 
    License 
     

🚀 Installation 
Prerequisites 

    Python 3.8 or higher
    4GB+ RAM (8GB+ recommended for large models)
    CUDA-compatible GPU (optional, for faster processing)
     

Quick Installation 
```bash
# Clone the repository
git clone https://github.com/SunnyThakur25/Layerguard-Enhanced.git
cd layerguard-enhanced

# Create virtual environment
python3 -m venv layerguard_env
source layerguard_env/bin/activate  # On Windows: layerguard_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install torch numpy scipy scikit-learn matplotlib seaborn pandas tqdm click rich transformers datasets fpdf2 PyYAML

# Make the tool executable
chmod +x layerguard_enhanced.py

# Use the provided installation script
chmod +x install_layerguard_enhanced.sh
./install_layerguard_enhanced.sh
source layerguard_enhanced_env/bin/activate
```
🏃 Quick Start 

# Basic Analysis 

```bash
# Analyze a PyTorch model
python layerguard_enhanced.py /path/to/your/model.pth

# Analyze a HuggingFace model
python layerguard_enhanced.py /path/to/your/hf_model/

# High sensitivity detection
python layerguard_enhanced.py model.pth --sensitivity high

# Generate visualizations
python layerguard_enhanced.py model.pth --visualize
```
# Understanding Output
```
# Exit codes for automation
Exit Code 0: Low risk - Model safe
Exit Code 1: Medium risk - Manual review needed
Exit Code 2: High risk - Security threat detected
```


🔧 Advanced Usage 
# Layer Filtering 
```bash
# Analyze only attention layers
python layerguard_enhanced.py model.pth --layer-filter "attention.*weight"

# Analyze embedding layers only
python layerguard_enhanced.py model.pth --layer-filter "embeddings"

# Focus on suspicious layer names
python layerguard_enhanced.py model.pth --layer-filter "(backdoor|trigger|malicious)"
```
# Custom Sensitivity Levels
```bash
# Low sensitivity (fewer false positives)
python layerguard_enhanced.py model.pth --sensitivity low

# Medium sensitivity (balanced) - DEFAULT
python layerguard_enhanced.py model.pth --sensitivity medium

# High sensitivity (more thorough detection)
python layerguard_enhanced.py model.pth --sensitivity high
```
# Comprehensive Analysis with All Features
```
python layerguard_enhanced.py model.pth \
    --sensitivity high \
    --layer-filter ".*" \
    --output detailed_report.yaml \
    --format yaml \
    --visualize \
    --hash-check
```

# 📋 Command Line Options 
```bash
--sensitivity
	
-s
	
Detection sensitivity (low/medium/high)
	
medium
--output
	
-o
	
Output file path for detailed report
	
None
--format
	
-f
	
Report format (json/yaml/markdown)
	
json
--visualize
	
-v
	
Generate PNG/PDF visualizations
	
False
--hash-check
	
-c
	
Perform file integrity check
	
False
--layer-filter
	
-l
	
Regex pattern for layer filtering
	
None
--batch-process
	
-b
	
Batch process multiple models
	
None

```
🔄 Batch Processing 
# Creating Model List 
```bash
# Create a text file with model paths (one per line)
echo -e "/models/model1.pth\n/models/hf_bert/\n/models/gpt_model.bin" > model_list.txt


# Running Batch Analysis
# Process multiple models
python layerguard_enhanced.py dummy_path --batch-process model_list.txt

# Batch processing with advanced options
python layerguard_enhanced.py dummy_path \
    --batch-process model_list.txt \
    --sensitivity high \
    --visualize \
    --format markdown

```
  # Batch Output Structure
```
batch_reports/
├── report_0_model1.pth.json
├── report_1_hf_bert.json
├── report_2_gpt_model.bin.json
├── batch_summary.json
└── batch_summary.md
```
# 📄 Report Formats 
JSON Format (Default) 
```json
{
  "timestamp": "2024-01-15T14:30:22.123456",
  "model_path": "suspicious_model.pth",
  "sensitivity": "high",
  "layer_analysis": {
    "total_layers": 650,
    "analyzed_layers": 650
  },
  "poisoned_layers": [
    {
      "layer_name": "backdoor_trigger.weight",
      "risk_score": 9.2,
      "indicators": [
        {"type": "suspicious_layer_name", "severity": "high"},
        {"type": "low_entropy", "severity": "high"}
      ]
    }
  ],
  "risk_score": 6.8
}
```

# YAML Format
```json
timestamp: "2024-01-15T14:30:22.123456"
model_path: "suspicious_model.pth"
sensitivity: "high"
layer_analysis:
  total_layers: 650
  analyzed_layers: 650
poisoned_layers:
  - layer_name: "backdoor_trigger.weight"
    risk_score: 9.2
    indicators:
      - type: "suspicious_layer_name"
        severity: "high"
      - type: "low_entropy"
        severity: "high"
risk_score: 6.8
```
# Markdown Format

```json
# LayerGuard Enhanced Analysis Report

## Summary

- **Model Path**: suspicious_model.pth
- **Model Type**: pytorch_checkpoint
- **Layers Analyzed**: 650
- **Poisoned Layers**: 8
- **Overall Risk Score**: 6.80/10.0

## Top Poisoned Layers

| Layer Name | Risk Score | Indicators |
|------------|------------|------------|
| backdoor_trigger.weight | 9.20 | suspicious_layer_name, low_entropy |
```

# 📊 Visualization Features 
Generated Charts 

    Layer Statistics Distribution (layer_statistics.png/.pdf) 
        Mean, standard deviation, entropy distributions
        Sparsity and kurtosis visualizations
         

    Risk Score Ranking (poisoned_layers_risk.png/.pdf) 
        Horizontal bar chart of top risky layers
        Color-coded by risk severity
         

    Correlation Heatmap (correlation_heatmap.png/.pdf) 
        Statistical relationships between layer properties
        Helps identify anomalous patterns
         
     

Visualization Benefits 

    Professional Reports: High-quality PDFs for presentations
    Stakeholder Communication: Clear visual evidence of threats
    Forensic Documentation: Comprehensive analysis records
    Pattern Recognition: Visual identification of anomalies
     

🏗️ Technical Architecture 
Core Components 
1. Model Loader

 # Supports multiple model formats
- PyTorch state dicts (.pth, .pt, .bin)
- HuggingFace Transformers models
- TensorFlow checkpoints (.h5)
- Custom model objects

  2. Statistical Analyzer
     # Advanced statistical methods
- Differential entropy calculation
- Isolation Forest anomaly detection
- Z-score analysis
- Kurtosis and skewness detection

  3. Pattern Recognizer
     # Suspicious pattern detection
- Layer name keyword matching
- Value distribution analysis
- Statistical outlier identification
- Cross-layer correlation analysis

4. Risk Scoring Engine
   # Weighted risk assessment
- Indicator severity weighting
- Layer-level risk aggregation
- Overall model risk calculation
- Dynamic threshold adjustment

  # Performance Optimizations 
Memory Management

# Efficient memory usage
- Large tensor sampling (100K+ elements)
- Progressive garbage collection
- Batch processing for large models
- GPU acceleration support

# Computational Efficiency
# Optimized processing
- Parallel layer analysis
- Cached statistical computations
- Early termination for obvious threats
- Progress tracking for long analyses

🔍 Detection Methods 
1. Statistical Anomaly Detection 

    Isolation Forest: Multivariate outlier detection
    Z-Score Analysis: Univariate statistical anomalies
    Entropy Analysis: Distribution uniformity detection
    Correlation Analysis: Cross-layer pattern recognition
     

2. Pattern Recognition 

    Suspicious Layer Names: Keyword-based detection
    Value Distribution Anomalies: Uniform/spiky distributions
    Sparsity Patterns: Abnormal zero-value ratios
    Statistical Moments: Kurtosis and skewness outliers
     

3. Risk Scoring System
   # Weighted scoring methodology
High Severity: 3.0 points
Medium Severity: 1.5 points  
Low Severity: 0.5 points

# Risk Categories
0.0-3.0: Low Risk (Normal)
3.1-6.0: Medium Risk (Review Needed)
6.1-10.0: High Risk (Security Threat)

4. Poison Indicators Detected 

    Extreme Parameter Values - Unusually large means/std deviations
    Low Differential Entropy - Uniform or repetitive weight patterns
    Excessive Zeros - Abnormal sparsity patterns
    Low Diversity - Limited unique parameter values
    Suspicious Layer Names - Keywords indicating backdoors
    Statistical Outliers - Layers that deviate from normal patterns
    Distribution Anomalies - Non-normal parameter distributions
    Correlation Abnormalities - Unexpected layer relationships
     

🛠️ Integration Examples 
CI/CD Pipeline Integration 
# GitHub Actions workflow
```
name: Model Security Check
on: [push, pull_request]

jobs:
  security-check:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Install LayerGuard
      run: |
        pip install torch numpy scipy scikit-learn matplotlib seaborn pandas tqdm click rich transformers fpdf2 PyYAML
    - name: Security Analysis
      run: |
        python layerguard_enhanced.py ${{ secrets.MODEL_PATH }} --sensitivity high
      if: always()

```

Enterprise Security Pipeline
```python
#!/bin/bash
# model_security_check.sh

MODELS_DIR="/models/to/analyze"
REPORT_DIR="/security/reports/$(date +%Y%m%d)"

# Batch process all models
find $MODELS_DIR -name "*.pth" -o -name "*.bin" -type f > model_list.txt
python layerguard_enhanced.py dummy --batch-process model_list.txt \
    --sensitivity high --visualize --format markdown \
    --output $REPORT_DIR/batch_report.json

# Check results and alert
HIGH_RISK_COUNT=$(grep -c '"risk_score": [8-9]' $REPORT_DIR/batch_report.json)
if [ $HIGH_RISK_COUNT -gt 0 ]; then
    echo "🚨 $HIGH_RISK_COUNT high-risk models detected!" | mail -s "Security Alert" security@company.com
    exit 2
fi🎯
```
Use Cases 
```bash
1. Third-Party Model Verification
   # Verify vendor-provided models
python layerguard_enhanced.py vendor_model.pth --sensitivity high --hash-check

2. Model Supply Chain Security
   # Check models before deployment
python layerguard_enhanced.py --batch-process production_models.txt --format yaml

3. Research and Development
   # Analyze experimental models
python layerguard_enhanced.py experimental_model.pth --layer-filter "new_layer.*" --visualize

4. Compliance and Auditing
   # Generate compliance reports
python layerguard_enhanced.py model.pth --format markdown --output compliance_report.md
```
🤝 Contributing 

We welcome contributions to LayerGuard Enhanced! Here's how you can help: 
Reporting Issues 

    Check existing issues before creating new ones
    Provide detailed reproduction steps
    Include model information and error messages
    Attach relevant log files
     

Feature Requests 

    Describe the use case clearly
    Explain the problem it solves
    Suggest implementation approach
    Consider backward compatibility
     

Code Contributions 

    Fork the repository
    Create a feature branch
    Implement your changes
    Add tests if applicable
    Update documentation
    Submit a pull request
     

Development Setup 
```
# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Code formatting
black layerguard_enhanced.py
flake8 layerguard_enhanced.py

# Type checking
mypy layerguard_enhanced.py
```
📚 Resources 
Documentation 

    AI Security Best Practices 
    Machine Learning Model Security 
    Adversarial Machine Learning 
     

Related Tools 

    ModelSanitizer  - Model integrity verification
    AISafetyToolkit  - Comprehensive AI safety tools
    NeuralDebugger  - Neural network debugging
     

Research Papers 

    "Detecting Backdoors in Neural Networks" - Chen et al., 2021
    "Adversarial Examples in ML" - Goodfellow et al., 2015
    "Model Poisoning Attacks" - Steinhardt et al., 2017
     

📞 Support 
Community Support 

    GitHub Issues: Report bugs and request features
    Discussions: Ask questions and share experiences
    Wiki: Documentation and tutorials
     

Professional Support 

For enterprise support, security consulting, or custom development: 

    Email: sunny48445@gmail.com 
    Commercial licensing available
    Priority response for enterprise customers
     

📄 License 

LayerGuard Enhanced is released under the MIT License. See the LICENSE  file for details. 

🙏 Acknowledgments 

    Thanks to the open-source community for inspiration
    Special thanks to researchers in AI security
    Built with ❤️ for the AI safety community



  <p align="center">
  <strong>Made with ❤️ for AI Security</strong>
</p>

<p align="center">
  <a href="https://github.com/SunnyThakur25/Layerguard-Enhanced">GitHub</a> •
  
</p>
 







