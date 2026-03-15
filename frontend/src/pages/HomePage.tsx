import { useEffect, useState, useRef } from "react";
import { createPrediction, getHistory, getDiseaseReport, type DiseaseReport } from "../api/predictionApi";
import { HistoryList } from "../components/HistoryList";
import { LoadingState } from "../components/LoadingState";
import { PredictionCard } from "../components/PredictionCard";
import { UploadForm } from "../components/UploadForm";
import type { PredictionRecord } from "../types/prediction";
import { jsPDF } from "jspdf";

// ============================================
// PDF GENERATION UTILITY
// ============================================
function generatePDFReport(report: DiseaseReport): void {
  const doc = new jsPDF();
  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 20;
  const contentWidth = pageWidth - margin * 2;
  let yPos = margin;

  // Colors
  const primaryColor: [number, number, number] = [30, 58, 95];
  const accentColor: [number, number, number] = [224, 123, 83];
  const textColor: [number, number, number] = [30, 41, 59];
  const mutedColor: [number, number, number] = [100, 116, 139];

  // Helper function for adding text with word wrap
  function addWrappedText(text: string, fontSize: number, isBold: boolean = false): void {
    doc.setFontSize(fontSize);
    doc.setTextColor(...textColor);
    doc.setFont("helvetica", isBold ? "bold" : "normal");
    
    const lines = doc.splitTextToSize(text, contentWidth);
    lines.forEach((line: string) => {
      if (yPos > 270) {
        doc.addPage();
        yPos = margin;
      }
      doc.text(line, margin, yPos);
      yPos += fontSize * 0.5;
    });
  }

  function addSpacer(height: number): void {
    yPos += height;
    if (yPos > 270) {
      doc.addPage();
      yPos = margin;
    }
  }

  // Header - Logo and Title
  doc.setFillColor(...primaryColor);
  doc.rect(0, 0, pageWidth, 40, "F");
  
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(22);
  doc.setFont("helvetica", "bold");
  doc.text("Skin Disease Analysis Report", margin, 25);
  
  yPos = 55;

  // Report Date
  const date = new Date(report.created_at).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
  doc.setFontSize(10);
  doc.setTextColor(...mutedColor);
  doc.text(`Report Date: ${date}`, margin, yPos);
  yPos += 15;

  // Disease Name Header
  doc.setFillColor(...accentColor);
  doc.rect(margin, yPos, contentWidth, 12, "F");
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(14);
  doc.setFont("helvetica", "bold");
  doc.text(report.disease_name.toUpperCase(), margin + 5, yPos + 8);
  yPos += 20;

  // Description Section
  doc.setFontSize(14);
  doc.setTextColor(...primaryColor);
  doc.setFont("helvetica", "bold");
  doc.text("Overview", margin, yPos);
  addSpacer(8);
  
  addWrappedText(report.description, 11);
  addSpacer(10);

  // Symptoms Section
  doc.setFontSize(14);
  doc.setTextColor(...primaryColor);
  doc.text("Common Symptoms", margin, yPos);
  addSpacer(8);
  
  doc.setFontSize(11);
  doc.setTextColor(...textColor);
  report.symptoms.forEach((symptom: string) => {
    doc.setFillColor(248, 250, 252);
    doc.roundedRect(margin + 5, yPos - 4, contentWidth - 10, 8, 1, 1, "F");
    doc.setFont("helvetica", "normal");
    doc.text(`• ${symptom}`, margin + 10, yPos + 1);
    yPos += 10;
  });
  addSpacer(10);

  // Risk Factors Section
  doc.setFontSize(14);
  doc.setTextColor(...primaryColor);
  doc.setFont("helvetica", "bold");
  doc.text("Risk Factors", margin, yPos);
  addSpacer(8);
  
  doc.setFontSize(11);
  report.risk_factors.forEach((factor: string) => {
    doc.setFont("helvetica", "normal");
    doc.text(`• ${factor}`, margin + 5, yPos);
    yPos += 7;
  });
  addSpacer(10);

  // Recommendations Section
  doc.setFontSize(14);
  doc.setTextColor(...primaryColor);
  doc.setFont("helvetica", "bold");
  doc.text("Recommendations", margin, yPos);
  addSpacer(8);
  
  doc.setFontSize(11);
  report.recommendations.forEach((rec: string, index: number) => {
    if (yPos > 250) {
      doc.addPage();
      yPos = margin;
    }
    doc.setFillColor(240, 253, 244);
    doc.roundedRect(margin + 5, yPos - 4, contentWidth - 10, 10, 1, 1, "F");
    doc.setFont("helvetica", "bold");
    doc.setTextColor(34, 197, 94);
    doc.text(`${index + 1}.`, margin + 10, yPos + 1);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(...textColor);
    const recText = doc.splitTextToSize(rec, contentWidth - 20);
    doc.text(recText, margin + 18, yPos + 1);
    yPos += 10 * recText.length + 3;
  });
  addSpacer(10);

  // When to See Doctor Section
  if (yPos > 220) {
    doc.addPage();
    yPos = margin;
  }
  
  doc.setFillColor(254, 249, 195);
  doc.roundedRect(margin, yPos, contentWidth, 25, 2, 2, "F");
  
  doc.setFontSize(12);
  doc.setTextColor(180, 83, 9);
  doc.setFont("helvetica", "bold");
  doc.text("When to See a Doctor", margin + 5, yPos + 8);
  
  doc.setFontSize(10);
  doc.setFont("helvetica", "normal");
  const doctorText = doc.splitTextToSize(report.when_to_see_doctor, contentWidth - 10);
  doc.text(doctorText, margin + 5, yPos + 16);

  // Save the PDF
  const fileName = `skindoc_report_${report.prediction_id}_${new Date().toISOString().split('T')[0]}.pdf`;
  doc.save(fileName);
}

// ============================================
// NAVIGATION COMPONENT
// ============================================
function Navbar() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const navbarRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
      setIsMobileMenuOpen(false);
    }
  };

  return (
    <nav className={`navbar ${isScrolled ? "scrolled" : ""}`} ref={navbarRef}>
      <div className="navbar-container">
        <a href="#" className="navbar-brand">
          <span className="navbar-brand-icon">⚕</span>
          SkinDoc
        </a>

        <ul className={`navbar-menu ${isMobileMenuOpen ? "active" : ""}`}>
          <li><a href="#features" onClick={(e) => { e.preventDefault(); scrollToSection("features"); }}>Features</a></li>
          <li><a href="#how-it-works" onClick={(e) => { e.preventDefault(); scrollToSection("how-it-works"); }}>How It Works</a></li>
          <li><a href="#diseases" onClick={(e) => { e.preventDefault(); scrollToSection("diseases"); }}>Conditions</a></li>
          <li><a href="#app" onClick={(e) => { e.preventDefault(); scrollToSection("app"); }}>Detection</a></li>
          <li><a href="#app" className="navbar-cta" onClick={(e) => { e.preventDefault(); scrollToSection("app"); }}>Get Started</a></li>
        </ul>

        <button 
          className="navbar-toggle" 
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          aria-label="Toggle menu"
          aria-expanded={isMobileMenuOpen}
        >
          <span></span>
          <span></span>
          <span></span>
        </button>
      </div>
    </nav>
  );
}

// ============================================
// HERO SECTION
// ============================================
function HeroSection() {
  const scrollToApp = () => {
    const element = document.getElementById("app");
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <section className="hero" id="hero">
      <div className="hero-container">
        <div className="hero-content">
          <div className="hero-badge">
            <span className="hero-badge-dot"></span>
            AI-Powered Detection
          </div>
          <h1>
            Advanced Skin Analysis with <span>AI Technology</span>
          </h1>
          <p className="hero-description">
            Upload a photo of your skin condition and get instant AI-powered analysis. 
            Receive detailed reports with symptoms, risk factors, and personalized recommendations for your skin health.
          </p>
          
          <div className="hero-stats">
            <div className="hero-stat">
              <div className="hero-stat-value">95%+</div>
              <div className="hero-stat-label">Accuracy Rate</div>
            </div>
            <div className="hero-stat">
              <div className="hero-stat-value">10K+</div>
              <div className="hero-stat-label">Images Analyzed</div>
            </div>
            <div className="hero-stat">
              <div className="hero-stat-value">7+</div>
              <div className="hero-stat-label">Conditions Detected</div>
            </div>
          </div>

          <div className="hero-actions">
            <button className="btn btn-accent btn-lg" onClick={scrollToApp}>
              Start Analysis
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M5 12h14M12 5l7 7-7 7"/>
              </svg>
            </button>
            <button className="btn btn-outline btn-lg" onClick={() => document.getElementById("how-it-works")?.scrollIntoView({ behavior: "smooth" })}>
              Learn More
            </button>
          </div>
        </div>

        <div className="hero-visual">
          <div className="hero-image-wrapper">
            <div className="hero-image" style={{ 
              background: "linear-gradient(135deg, #1e3a5f 0%, #2d5a8a 50%, #4a7eb8 100%)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexDirection: "column",
              color: "white"
            }}>
              <svg width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ opacity: 0.9 }}>
                <path d="M9 12l2 2 4-4"/>
                <circle cx="12" cy="12" r="10"/>
                <path d="M12 8v4l2 2"/>
              </svg>
              <p style={{ marginTop: "1rem", fontSize: "1.125rem", opacity: 0.9 }}>AI Analysis Ready</p>
            </div>
            <div className="hero-image-overlay">
              <div className="hero-image-card">
                <div className="hero-image-card-icon">✓</div>
                <div className="hero-image-card-content">
                  <h4>Instant Analysis</h4>
                  <p>Results in seconds</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// ============================================
// FEATURES SECTION
// ============================================
function FeaturesSection() {
  const features = [
    {
      icon: "🔬",
      title: "Advanced AI Model",
      description: "State-of-the-art deep learning algorithms trained on extensive medical datasets for accurate skin condition detection."
    },
    {
      icon: "⚡",
      title: "Instant Results",
      description: "Get your analysis results in seconds. No waiting—just upload and receive immediate insights about your condition."
    },
    {
      icon: "🔒",
      title: "Privacy Protected",
      description: "Your health data is encrypted and secure. We prioritize your privacy with enterprise-grade security measures."
    },
    {
      icon: "📱",
      title: "Easy to Use",
      description: "Simple, intuitive interface. Just upload an image and get comprehensive results instantly."
    },
    {
      icon: "📊",
      title: "Detailed Reports",
      description: "Receive comprehensive reports with symptoms, risk factors, and personalized recommendations."
    },
    {
      icon: "🩺",
      title: "Professional Quality",
      description: "Built with medical-grade accuracy to provide reliable skin condition analysis."
    }
  ];

  return (
    <section className="features" id="features">
      <div className="section-header">
        <span className="section-label">Why Choose Us</span>
        <h2 className="section-title">Powerful Features for Accurate Detection</h2>
        <p className="section-description">
          Our platform combines cutting-edge AI technology with medical expertise to provide you with accurate skin disease detection.
        </p>
      </div>

      <div className="features-grid">
        {features.map((feature, index) => (
          <div className="feature-card" key={index}>
            <div className="feature-icon">{feature.icon}</div>
            <h3>{feature.title}</h3>
            <p>{feature.description}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

// ============================================
// HOW IT WORKS SECTION
// ============================================
function HowItWorksSection() {
  const steps = [
    {
      number: 1,
      title: "Upload Your Image",
      description: "Take a clear photo of the affected skin area or upload an existing image. Our system accepts all common image formats."
    },
    {
      number: 2,
      title: "AI Analysis",
      description: "Our advanced deep learning model analyzes the image against thousands of medical images to identify potential conditions."
    },
    {
      number: 3,
      title: "Get Your Results",
      description: "Receive instant results with detailed information, symptoms, risk factors, and recommendations for your care."
    }
  ];

  return (
    <section className="how-it-works" id="how-it-works">
      <div className="section-header">
        <span className="section-label">Process</span>
        <h2 className="section-title">How It Works</h2>
        <p className="section-description">
          Get accurate skin disease detection in just three simple steps.
        </p>
      </div>

      <div className="steps-container">
        <div className="steps">
          {steps.map((step, index) => (
            <div className="step" key={index}>
              <div className="step-number">{step.number}</div>
              <div className="step-content">
                <h3>{step.title}</h3>
                <p>{step.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ============================================
// DISEASES SECTION
// ============================================
function DiseasesSection() {
  const diseases = [
    { icon: "🦠", name: "Melanoma", description: "Skin cancer" },
    { icon: "🔴", name: "Melanocytic Nevi", description: "Moles" },
    { icon: "🏁", name: "Basal Cell Carcinoma", description: "BCC cancer" },
    { icon: "⚪", name: "Actinic Keratoses", description: "Pre-cancer" },
    { icon: "📛", name: "Benign Keratosis", description: "Non-cancerous" },
    { icon: "🔵", name: "Dermatofibroma", description: "Skin nodules" },
    { icon: "👽", name: "Vascular Lesions", description: "Blood vessels" },
    { icon: "🎗️", name: "Psoriasis", description: "Skin condition" }
  ];

  return (
    <section className="diseases" id="diseases">
      <div className="section-header">
        <span className="section-label">Conditions</span>
        <h2 className="section-title">Conditions We Detect</h2>
        <p className="section-description">
          Our AI system can identify a wide range of skin conditions, from common issues to serious medical conditions.
        </p>
      </div>

      <div className="diseases-grid">
        {diseases.map((disease, index) => (
          <div className="disease-card" key={index}>
            <div className="disease-icon">{disease.icon}</div>
            <h4>{disease.name}</h4>
            <p>{disease.description}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

// ============================================
// STATS SECTION
// ============================================
function StatsSection() {
  return (
    <section className="stats">
      <div className="stats-grid">
        <div className="stat-item">
          <h3>95%+</h3>
          <p>Detection Accuracy</p>
        </div>
        <div className="stat-item">
          <h3>10,000+</h3>
          <p>Images Analyzed</p>
        </div>
        <div className="stat-item">
          <h3>8</h3>
          <p>Conditions Detected</p>
        </div>
        <div className="stat-item">
          <h3>24/7</h3>
          <p>AI Availability</p>
        </div>
      </div>
    </section>
  );
}

// ============================================
// APP INTERFACE SECTION
// ============================================
function AppInterfaceSection() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionRecord | null>(null);
  const [history, setHistory] = useState<PredictionRecord[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoadingReport, setIsLoadingReport] = useState(false);

  useEffect(() => {
    void loadHistory();
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }

    const nextUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(nextUrl);

    return () => URL.revokeObjectURL(nextUrl);
  }, [selectedFile]);

  async function loadHistory() {
    try {
      const response = await getHistory();
      setHistory(response.items);
      if (!prediction && response.items.length > 0) {
        setPrediction(response.items[0]);
      }
    } catch (loadError) {
      console.error("Failed to load history:", loadError);
    }
  }

  async function handleSubmit() {
    if (!selectedFile) {
      setError("Select an image before starting detection.");
      return;
    }

    try {
      setError(null);
      setIsSubmitting(true);
      const result = await createPrediction(selectedFile);
      setPrediction(result);
      setHistory((current) => [result, ...current.filter((item) => item.id !== result.id)]);
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Detection failed.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleDownloadReport() {
    if (!prediction) return;

    try {
      setIsLoadingReport(true);
      const report = await getDiseaseReport(prediction.id);
      generatePDFReport(report);
    } catch (err) {
      console.error("Failed to generate report:", err);
      setError("Failed to generate report. Please try again.");
    } finally {
      setIsLoadingReport(false);
    }
  }

  return (
    <section className="app-interface" id="app">
      <div className="section-header">
        <span className="section-label">Try It Now</span>
        <h2 className="section-title">Start Your Analysis</h2>
        <p className="section-description">
          Upload a skin image and get instant AI-powered analysis with detailed PDF reports.
        </p>
      </div>

      <div className="app-shell">
        {error ? <div className="error-banner">{error}</div> : null}
        {isSubmitting ? <LoadingState /> : null}

        <div className="content-grid">
          <UploadForm
            previewUrl={previewUrl}
            isSubmitting={isSubmitting}
            onFileChange={setSelectedFile}
            onSubmit={handleSubmit}
          />
          <div className="panel result-panel">
            <PredictionCard prediction={prediction} />
            {prediction && (
              <button 
                className="btn btn-primary" 
                style={{ marginTop: "1rem", width: "100%" }}
                onClick={handleDownloadReport}
                disabled={isLoadingReport}
              >
                {isLoadingReport ? (
                  <>
                    <span className="loading-orb" style={{ display: "inline-block", marginRight: "8px" }}></span>
                    Generating Report...
                  </>
                ) : (
                  <>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ marginRight: "8px" }}>
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                      <polyline points="14 2 14 8 20 8"></polyline>
                      <line x1="12" y1="18" x2="12" y2="12"></line>
                      <line x1="9" y1="15" x2="12" y2="18"></line>
                      <line x1="15" y1="15" x2="12" y2="18"></line>
                    </svg>
                    Download PDF Report
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        <HistoryList items={history} onSelect={setPrediction} />
      </div>
    </section>
  );
}

// ============================================
// FOOTER
// ============================================
function Footer() {
  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-grid">
          <div className="footer-about">
            <div className="footer-brand">
              <span className="footer-brand-icon">⚕</span>
              SkinDoc
            </div>
            <p>
              Advanced AI-powered skin disease detection system. Our mission is to make early skin disease detection accessible to everyone.
            </p>
            <div className="footer-social">
              <a href="#" aria-label="Twitter">𝕏</a>
              <a href="#" aria-label="Facebook">f</a>
              <a href="#" aria-label="LinkedIn">in</a>
              <a href="#" aria-label="Instagram">📷</a>
            </div>
          </div>

          <div className="footer-column">
            <h4>Product</h4>
            <ul className="footer-links">
              <li><a href="#features">Features</a></li>
              <li><a href="#how-it-works">How It Works</a></li>
              <li><a href="#diseases">Conditions</a></li>
              <li><a href="#app">Detection Tool</a></li>
            </ul>
          </div>

          <div className="footer-column">
            <h4>Company</h4>
            <ul className="footer-links">
              <li><a href="#">About Us</a></li>
              <li><a href="#">Contact</a></li>
              <li><a href="#">Privacy Policy</a></li>
              <li><a href="#">Terms of Service</a></li>
            </ul>
          </div>

          <div className="footer-column">
            <h4>Resources</h4>
            <ul className="footer-links">
              <li><a href="#">Help Center</a></li>
              <li><a href="#">Documentation</a></li>
              <li><a href="#">API</a></li>
              <li><a href="#">Status</a></li>
            </ul>
          </div>
        </div>

        <div className="footer-bottom">
          <p>© 2024 SkinDoc. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
}

// ============================================
// MAIN HOMEPAGE COMPONENT
// ============================================
export function HomePage() {
  return (
    <>
      <Navbar />
      <main>
        <HeroSection />
        <FeaturesSection />
        <HowItWorksSection />
        <DiseasesSection />
        <StatsSection />
        <AppInterfaceSection />
      </main>
      <Footer />
    </>
  );
}
