/**
 * NeuroScan AI - ADHD Detection Frontend
 * Modular ES6+ JavaScript with Clean Architecture
 *
 * @module NeuroScanApp
 * @version 1.0.0
 */

// ============================================================
// Configuration Module
// ============================================================
const Config = Object.freeze({
  API_DELAY: 2000,
  MAX_FILE_SIZE: 50 * 1024 * 1024, // 50MB
  ALLOWED_EXTENSIONS: [".csv", ".txt", ".edf"],
  MODEL_INFO: {
    channels: 19,
    features: 65,
    model: "Ensemble",
    accuracy: 70.63,
  },
  EEG_CHANNELS: [
    "fp1",
    "fp2",
    "f3",
    "f4",
    "c3",
    "c4",
    "p3",
    "p4",
    "o1",
    "o2",
    "f7",
    "f8",
    "t7",
    "t8",
    "p7",
    "p8",
    "fz",
    "cz",
    "pz",
  ],
  BEHAVIORAL_FIELDS: [],
});

// ============================================================
// State Management Module
// ============================================================
const State = {
  file: null,
  isProcessing: false,
  results: null,
  inputMode: "upload", // 'upload' or 'manual'
  manualData: null,

  reset() {
    this.file = null;
    this.isProcessing = false;
    this.results = null;
    this.manualData = null;
  },
};

// ============================================================
// DOM References Module
// ============================================================
const DOM = {
  // Form Elements
  form: () => document.getElementById("analysis-form"),
  dropzone: () => document.getElementById("dropzone"),
  fileInput: () => document.getElementById("file-input"),
  filePreview: () => document.getElementById("file-preview"),
  fileName: () => document.getElementById("file-name"),
  fileSize: () => document.getElementById("file-size"),
  removeFileBtn: () => document.getElementById("remove-file"),
  patientId: () => document.getElementById("patient-id"),
  patientAge: () => document.getElementById("patient-age"),
  submitBtn: () => document.getElementById("submit-btn"),
  formError: () => document.getElementById("form-error"),
  errorMessage: () => document.getElementById("error-message"),

  // Input Mode Toggle
  modeUploadBtn: () => document.getElementById("mode-upload"),
  modeManualBtn: () => document.getElementById("mode-manual"),
  uploadSection: () => document.getElementById("upload-section"),
  manualSection: () => document.getElementById("manual-section"),

  // Manual Input Elements
  fillSampleBtn: () => document.getElementById("fill-sample-btn"),
  clearInputsBtn: () => document.getElementById("clear-inputs-btn"),

  // EEG Input Fields
  getEegInput: (channel) => document.getElementById(`eeg-${channel}`),

  // Results Elements
  resultsSection: () => document.getElementById("results-section"),
  resultsTimestamp: () => document.getElementById("results-timestamp"),
  predictionCard: () => document.getElementById("prediction-card"),
  resultIcon: () => document.getElementById("result-icon"),
  predictionResult: () => document.getElementById("prediction-result"),
  confidenceValue: () => document.getElementById("confidence-value"),
  confidenceFill: () => document.getElementById("confidence-fill"),

  // Stats
  statChannels: () => document.getElementById("stat-channels"),
  statDatapoints: () => document.getElementById("stat-datapoints"),
  statFeatures: () => document.getElementById("stat-features"),
  statModel: () => document.getElementById("stat-model"),

  // Action Buttons
  newAnalysisBtn: () => document.getElementById("new-analysis-btn"),
  downloadBtn: () => document.getElementById("download-btn"),
  exportBtn: () => document.getElementById("export-btn"),
};

// ============================================================
// Utility Functions Module
// ============================================================
const Utils = {
  /**
   * Format bytes to human-readable string
   * @param {number} bytes - File size in bytes
   * @returns {string} Formatted size string
   */
  formatFileSize(bytes) {
    if (bytes === 0) return "0 Bytes";
    const units = ["Bytes", "KB", "MB", "GB"];
    const k = 1024;
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${units[i]}`;
  },

  /**
   * Get current timestamp formatted
   * @returns {string} Formatted timestamp
   */
  getTimestamp() {
    return new Date().toLocaleString("en-US", {
      dateStyle: "medium",
      timeStyle: "short",
    });
  },

  /**
   * Extract file extension
   * @param {string} filename - File name
   * @returns {string} Extension with dot
   */
  getExtension(filename) {
    return "." + filename.split(".").pop().toLowerCase();
  },

  /**
   * Generate random number in range
   * @param {number} min - Minimum value
   * @param {number} max - Maximum value
   * @returns {number} Random integer
   */
  randomInRange(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  },

  /**
   * Delay execution
   * @param {number} ms - Milliseconds to wait
   * @returns {Promise} Promise that resolves after delay
   */
  delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  },
};

// ============================================================
// Validation Module
// ============================================================
const Validator = {
  /**
   * Validate uploaded file
   * @param {File} file - File to validate
   * @returns {Object} Validation result
   */
  validateFile(file) {
    if (!file) {
      return { isValid: false, message: "Please select a file to upload" };
    }

    const extension = Utils.getExtension(file.name);
    if (!Config.ALLOWED_EXTENSIONS.includes(extension)) {
      return {
        isValid: false,
        message: `Invalid file type. Supported: ${Config.ALLOWED_EXTENSIONS.join(", ")}`,
      };
    }

    if (file.size > Config.MAX_FILE_SIZE) {
      return {
        isValid: false,
        message: "File size exceeds 50MB limit",
      };
    }

    return { isValid: true, message: "" };
  },

  /**
   * Validate manual EEG input
   * @returns {Object} Validation result with data
   */
  validateManualInput() {
    const data = {};
    const missingFields = [];

    // Validate behavioral fields
    for (const field of Config.BEHAVIORAL_FIELDS) {
      const input = DOM.getEegInput(field);
      const value = input?.value?.trim();
      if (!value || isNaN(parseFloat(value))) {
        missingFields.push(field.charAt(0).toUpperCase() + field.slice(1));
      } else {
        data[field] = parseFloat(value);
      }
    }

    // Validate EEG channels
    for (const channel of Config.EEG_CHANNELS) {
      const input = DOM.getEegInput(channel);
      const value = input?.value?.trim();
      if (!value || isNaN(parseFloat(value))) {
        missingFields.push(channel.toUpperCase());
      } else {
        data[channel] = parseFloat(value);
      }
    }

    if (missingFields.length > 0) {
      const displayFields =
        missingFields.length > 5
          ? missingFields.slice(0, 5).join(", ") +
            ` and ${missingFields.length - 5} more`
          : missingFields.join(", ");
      return {
        isValid: false,
        message: `Please fill in all required fields: ${displayFields}`,
        data: null,
      };
    }

    return { isValid: true, message: "", data };
  },

  /**
   * Validate form before submission
   * @returns {Object} Validation result
   */
  validateForm() {
    if (State.inputMode === "manual") {
      return this.validateManualInput();
    }

    if (!State.file) {
      return {
        isValid: false,
        message: "Please upload an EEG file to continue",
      };
    }
    return this.validateFile(State.file);
  },
};

// ============================================================
// UI Module
// ============================================================
const UI = {
  /**
   * Show file preview
   * @param {File} file - Uploaded file
   */
  showFilePreview(file) {
    DOM.fileName().textContent = file.name;
    DOM.fileSize().textContent = Utils.formatFileSize(file.size);
    DOM.filePreview().classList.remove("hidden");
    DOM.dropzone().classList.add("hidden");
  },

  /**
   * Hide file preview and show dropzone
   */
  hideFilePreview() {
    DOM.filePreview().classList.add("hidden");
    DOM.dropzone().classList.remove("hidden");
    DOM.fileInput().value = "";
  },

  /**
   * Switch input mode
   * @param {string} mode - 'upload' or 'manual'
   */
  switchInputMode(mode) {
    State.inputMode = mode;

    // Update toggle buttons
    DOM.modeUploadBtn().classList.toggle("active", mode === "upload");
    DOM.modeManualBtn().classList.toggle("active", mode === "manual");

    // Show/hide sections
    DOM.uploadSection().classList.toggle("hidden", mode !== "upload");
    DOM.manualSection().classList.toggle("hidden", mode !== "manual");

    // Clear any errors
    this.hideError();
  },

  /**
   * Fill sample EEG data from actual dataset
   */
  async fillSampleData() {
    try {
      UI.showLoading("Loading sample data from dataset...");
      const response = await fetch("/api/sample-data");
      const data = await response.json();

      if (data.success && data.sample) {
        // Fill all EEG fields with actual dataset values
        for (const [key, value] of Object.entries(data.sample)) {
          const input = DOM.getEegInput(key.toLowerCase());
          if (input) {
            input.value = parseFloat(value).toFixed(2);
          }
        }
        UI.showInfo("Sample data loaded from dataset successfully");
      } else {
        throw new Error(data.error || "Failed to load sample data");
      }
    } catch (error) {
      UI.showError(`Error loading sample data: ${error.message}`);
    }
  },

  /**
   * Clear all manual input fields
   */
  clearManualInputs() {
    // Clear behavioral fields
    for (const field of Config.BEHAVIORAL_FIELDS) {
      const input = DOM.getEegInput(field);
      if (input) input.value = "";
    }

    // Clear EEG channels
    for (const channel of Config.EEG_CHANNELS) {
      const input = DOM.getEegInput(channel);
      if (input) input.value = "";
    }
  },

  /**
   * Show form error
   * @param {string} message - Error message
   */
  showError(message) {
    DOM.errorMessage().textContent = message;
    DOM.formError().classList.remove("hidden");
  },

  /**
   * Hide form error
   */
  hideError() {
    DOM.formError().classList.add("hidden");
  },

  /**
   * Show loading message
   * @param {string} message - Loading message
   */
  showLoading(message) {
    DOM.errorMessage().textContent = message;
    DOM.formError().className = "form-error success";
    DOM.formError().classList.remove("hidden");
  },

  /**
   * Hide loading message
   */
  hideLoading() {
    DOM.formError().classList.add("hidden");
  },

  /**
   * Show success message
   * @param {string} message - Success message
   */
  showSuccess(message) {
    DOM.errorMessage().textContent = message;
    DOM.formError().className = "form-error success";
    DOM.formError().classList.remove("hidden");

    // Auto-hide after 3 seconds
    setTimeout(() => {
      this.hideError();
    }, 3000);
  },

  /**
   * Show info message
   * @param {string} message - Info message
   */
  showInfo(message) {
    DOM.errorMessage().textContent = message;
    DOM.formError().className = "form-error success";
    DOM.formError().classList.remove("hidden");

    // Auto-hide after 3 seconds
    setTimeout(() => {
      this.hideError();
    }, 3000);
  },

  /**
   * Set button loading state
   * @param {boolean} isLoading - Loading state
   */
  setButtonLoading(isLoading) {
    const btn = DOM.submitBtn();
    if (isLoading) {
      btn.classList.add("loading");
      btn.disabled = true;
    } else {
      btn.classList.remove("loading");
      btn.disabled = false;
    }
  },

  /**
   * Add drag-over class to dropzone
   */
  addDragOver() {
    DOM.dropzone().classList.add("drag-over");
  },

  /**
   * Remove drag-over class from dropzone
   */
  removeDragOver() {
    DOM.dropzone().classList.remove("drag-over");
  },

  /**
   * Display analysis results
   * @param {Object} results - Analysis results
   */
  displayResults(results) {
    State.results = results;
    const { classification, classType, confidence, dataPoints } = results;

    // Update timestamp
    DOM.resultsTimestamp().textContent = Utils.getTimestamp();

    // Update prediction card
    const card = DOM.predictionCard();
    card.classList.remove("adhd", "control");
    card.classList.add(classType);

    // Update result icon
    DOM.resultIcon().innerHTML = Icons.getResultIcon(classType);

    // Update prediction text
    DOM.predictionResult().textContent = classification;

    // Update confidence
    DOM.confidenceValue().textContent = `${confidence}%`;

    // Animate confidence fill with delay
    requestAnimationFrame(() => {
      DOM.confidenceFill().style.width = `${confidence}%`;
    });

    // Update stats
    DOM.statChannels().textContent = Config.MODEL_INFO.channels;
    DOM.statDatapoints().textContent = dataPoints;
    DOM.statFeatures().textContent = Config.MODEL_INFO.features;
    DOM.statModel().textContent = Config.MODEL_INFO.model;

    // Show results section
    DOM.resultsSection().classList.remove("hidden");

    // Scroll to results
    setTimeout(() => {
      DOM.resultsSection().scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }, 100);
  },

  /**
   * Hide results and reset form
   */
  resetUI() {
    State.reset();
    this.hideFilePreview();
    this.hideError();
    this.setButtonLoading(false);
    DOM.resultsSection().classList.add("hidden");
    DOM.confidenceFill().style.width = "0%";
    DOM.patientId().value = "";
    DOM.patientAge().value = "";
    this.clearManualInputs();
    this.switchInputMode("upload");
    window.scrollTo({ top: 0, behavior: "smooth" });
  },
};

// ============================================================
// Icons Module
// ============================================================
const Icons = {
  /**
   * Get result icon SVG based on classification
   * @param {string} type - 'adhd' or 'control'
   * @returns {string} SVG markup
   */
  getResultIcon(type) {
    if (type === "adhd") {
      return `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                    <line x1="12" y1="9" x2="12" y2="13"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
            `;
    }
    return `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                <polyline points="22,4 12,14.01 9,11.01"/>
            </svg>
        `;
  },
};

// ============================================================
// Analysis Module (Simulated AI)
// ============================================================
const Analyzer = {
  /**
   * Call API endpoint for EEG analysis
   * @param {Object} manualData - Optional manual input data
   * @returns {Promise<Object>} Analysis results
   */
  async analyze(manualData = null) {
    try {
      const formData = new FormData();

      if (manualData) {
        // Manual input mode
        formData.append("mode", "manual");
        formData.append("age", manualData.age || 0);
        formData.append("attention_score", manualData.attention || 0);
        formData.append("hyperactivity_score", manualData.hyperactivity || 0);
        formData.append("impulsivity_score", manualData.impulsivity || 0);

        for (const channel of Config.EEG_CHANNELS) {
          const value = manualData[channel.toLowerCase()] || 0;
          formData.append(channel.toLowerCase(), value);
        }
      } else {
        // File upload mode
        formData.append("mode", "upload");
        if (State.file) {
          formData.append("file", State.file);
        }
      }

      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || "API error");
      }

      const result = await response.json();

      if (result.success) {
        return {
          classification: result.classification,
          classType: result.classType,
          confidence: result.confidence,
          dataPoints: result.dataPoints,
          timestamp: result.timestamp,
          inputMode: result.inputMode || (manualData ? "manual" : "file"),
        };
      } else {
        throw new Error(result.error || "Analysis failed");
      }
    } catch (error) {
      console.error("API Error:", error);
      throw error;
    }
  },
};

// ============================================================
// Report Generator Module
// ============================================================
const ReportGenerator = {
  /**
   * Generate and download PDF clinical report
   */
  async generateReport() {
    if (!State.results) {
      UI.showError(
        "No analysis results available. Please run an analysis first.",
      );
      return;
    }

    try {
      UI.showLoading("Generating PDF clinical report...");

      const { classification, confidence, dataPoints, inputMode } =
        State.results;
      const patientId = DOM.patientId().value || "Unknown";
      const patientAge = DOM.patientAge().value || "Not provided";

      // Prepare EEG data for report
      const eegData = {};
      if (State.manualData && inputMode === "manual") {
        for (const [key, value] of Object.entries(State.manualData)) {
          if (Config.EEG_CHANNELS.includes(key.toUpperCase())) {
            eegData[key] = value;
          }
        }
      }

      // Prepare report data
      const reportData = {
        patientId: patientId,
        patientAge: patientAge,
        classification: classification,
        confidence: confidence,
        dataPoints: dataPoints,
        timestamp: new Date().toISOString(),
        inputMethod: inputMode,
        eegData: eegData,
      };

      // Send to backend for PDF generation
      const response = await fetch("/api/generate-report", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(reportData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.error || `HTTP error! status: ${response.status}`,
        );
      }

      // Get PDF blob and download
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `ADHD_Assessment_${Date.now()}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      UI.hideLoading();
      UI.showSuccess("Clinical report downloaded successfully!");
    } catch (error) {
      console.error("PDF generation error:", error);
      UI.hideLoading();
      UI.showError(`Failed to generate report: ${error.message}`);
    }
  },

  /**
   * Alternative: Generate downloadable text report (fallback)
   */
  downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  },
};

// ============================================================
// Event Handlers Module
// ============================================================
const EventHandlers = {
  /**
   * Handle input mode toggle
   * @param {string} mode - 'upload' or 'manual'
   */
  onModeToggle(mode) {
    UI.switchInputMode(mode);
  },

  /**
   * Handle fill sample data button
   */
  async onFillSampleData() {
    await UI.fillSampleData();
  },

  /**
   * Handle clear inputs button
   */
  onClearInputs() {
    UI.clearManualInputs();
  },

  /**
   * Handle file selection
   * @param {File} file - Selected file
   */
  handleFileSelect(file) {
    const validation = Validator.validateFile(file);

    if (!validation.isValid) {
      UI.showError(validation.message);
      return;
    }

    UI.hideError();
    State.file = file;
    UI.showFilePreview(file);
  },

  /**
   * Handle file input change
   * @param {Event} event - Change event
   */
  onFileInputChange(event) {
    const file = event.target.files[0];
    if (file) {
      this.handleFileSelect(file);
    }
  },

  /**
   * Handle drag over
   * @param {DragEvent} event - Drag event
   */
  onDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    UI.addDragOver();
  },

  /**
   * Handle drag leave
   * @param {DragEvent} event - Drag event
   */
  onDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    UI.removeDragOver();
  },

  /**
   * Handle file drop
   * @param {DragEvent} event - Drop event
   */
  onDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    UI.removeDragOver();

    const files = event.dataTransfer.files;
    if (files.length > 0) {
      this.handleFileSelect(files[0]);
    }
  },

  /**
   * Handle dropzone keyboard navigation
   * @param {KeyboardEvent} event - Keyboard event
   */
  onDropzoneKeyDown(event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      DOM.fileInput().click();
    }
  },

  /**
   * Handle remove file button click
   */
  onRemoveFile() {
    State.file = null;
    UI.hideFilePreview();
    UI.hideError();
  },

  /**
   * Handle form submission
   * @param {Event} event - Submit event
   */
  async onFormSubmit(event) {
    event.preventDefault();

    if (State.isProcessing) return;

    // Validate form
    const validation = Validator.validateForm();
    if (!validation.isValid) {
      UI.showError(validation.message);
      return;
    }

    UI.hideError();
    State.isProcessing = true;
    UI.setButtonLoading(true);

    try {
      let results;
      if (State.inputMode === "manual") {
        State.manualData = validation.data;
        results = await Analyzer.analyze(validation.data);
      } else {
        results = await Analyzer.analyze();
      }
      UI.displayResults(results);
    } catch (error) {
      UI.showError("Analysis failed. Please try again.");
      console.error("Analysis error:", error);
    } finally {
      State.isProcessing = false;
      UI.setButtonLoading(false);
    }
  },

  /**
   * Handle new analysis button click
   */
  onNewAnalysis() {
    UI.resetUI();
  },

  /**
   * Handle download report button click
   */
  onDownloadReport() {
    ReportGenerator.generateReport();
  },
};

// ============================================================
// Application Initialization
// ============================================================
const App = {
  /**
   * Initialize event listeners
   */
  initEventListeners() {
    // Input mode toggle
    DOM.modeUploadBtn().addEventListener("click", () =>
      EventHandlers.onModeToggle("upload"),
    );
    DOM.modeManualBtn().addEventListener("click", () =>
      EventHandlers.onModeToggle("manual"),
    );

    // Manual input buttons
    DOM.fillSampleBtn().addEventListener("click", async () =>
      EventHandlers.onFillSampleData(),
    );
    DOM.clearInputsBtn().addEventListener("click", () =>
      EventHandlers.onClearInputs(),
    );

    // File input
    DOM.fileInput().addEventListener("change", (e) =>
      EventHandlers.onFileInputChange(e),
    );

    // Dropzone
    const dropzone = DOM.dropzone();
    dropzone.addEventListener("dragover", (e) => EventHandlers.onDragOver(e));
    dropzone.addEventListener("dragleave", (e) => EventHandlers.onDragLeave(e));
    dropzone.addEventListener("drop", (e) => EventHandlers.onDrop(e));
    dropzone.addEventListener("keydown", (e) =>
      EventHandlers.onDropzoneKeyDown(e),
    );

    // Remove file button
    DOM.removeFileBtn().addEventListener("click", () =>
      EventHandlers.onRemoveFile(),
    );

    // Form submission
    DOM.form().addEventListener("submit", (e) => EventHandlers.onFormSubmit(e));

    // Results actions
    DOM.newAnalysisBtn().addEventListener("click", () =>
      EventHandlers.onNewAnalysis(),
    );
    DOM.downloadBtn().addEventListener("click", () =>
      EventHandlers.onDownloadReport(),
    );
    DOM.exportBtn().addEventListener("click", () =>
      EventHandlers.onDownloadReport(),
    );
  },

  /**
   * Initialize application
   */
  init() {
    console.log("🧠 NeuroScan AI v1.0.0 initialized");
    this.initEventListeners();
  },
};

// ============================================================
// Bootstrap Application
// ============================================================
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => App.init());
} else {
  App.init();
}

// Export for module usage (optional)
export { App, State, Config, Utils };
