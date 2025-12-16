import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, History, LogOut, Loader2, CheckCircle, 
  AlertCircle, ChevronDown, ChevronUp, BarChart3, Image as ImageIcon,
  Shield, Zap, Brain, Menu, X, User, Lock, Camera, 
  LayoutDashboard, FileText, ScanLine, Clock, Calendar, Users, Mail, ArrowLeft, Key, Trash2,
  ArrowRight, Layers, Cpu, Play, MonitorPlay, BookOpen, Target, Sparkles, Database, File, Download, Eye,
  Github, Linkedin, ScanEye
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
  PieChart, Pie, Cell
} from 'recharts';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

// --- UTILS ---
function cn(...inputs) {
  return twMerge(clsx(inputs));
}

// Configuration
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

// --- CONSTANTS ---
const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
const ALLOWED_FILE_TYPES = ['image/jpeg', 'image/png', 'image/jpg'];

const validateFile = (file) => {
  if (!ALLOWED_FILE_TYPES.includes(file.type)) {
    return "Invalid file type. Only JPG, PNG allowed.";
  }
  if (file.size > MAX_FILE_SIZE) {
    return "File too large. Max 5MB.";
  }
  return null;
};

// --- DATASET & MODEL DATA ---

const MODEL_PERFORMANCE = [
  { name: 'Vanilla', Accuracy: 0.857, Precision: 0.870, Recall: 0.857, F1: 0.861 },
  { name: 'Distilled', Accuracy: 0.952, Precision: 0.960, Recall: 0.952, F1: 0.954 },
  { name: 'AKTP', Accuracy: 1.000, Precision: 1.000, Recall: 1.000, F1: 1.000 },
];

const CLASS_DISTRIBUTION = [
  { name: 'Angle Change', value: 29, color: '#6366f1' }, // Indigo
  { name: 'Dark/Flash', value: 26, color: '#8b5cf6' },   // Violet
  { name: 'Normal', value: 23, color: '#ec4899' },       // Pink
  { name: 'Negative', value: 23, color: '#64748b' },     // Slate
];

const MODEL_SPECS = [
  { label: "Student Architecture", value: "SVM (Support Vector Machine)" },
  { label: "Teacher Architecture", value: "ResNet18 + EfficientNet-B0" },
  { label: "Feature Extraction", value: "Harris-SIFT" },
  { label: "Distillation Method", value: "AKTP (Adaptive Knowledge Transfer)" },
  { label: "Kernel Type", value: "RBF (Radial Basis Function)" },
  { label: "Distillation Temp", value: "T=4.0" },
  { label: "Loss Weight (Alpha)", value: "0.7" },
  { label: "Dataset Size", value: "101 Images (224x224px)" },
  { label: "Classification", value: "Binary (Positive/Negative)" },
];

const TEAM_MEMBERS = [
  { 
    name: "Stanley Pratama Teguh", 
    id: "2702311566", 
    role: "Full Stack Developer", 
    image: "/stanley.jpg" 
  },
  { 
    name: "Gading Aditya Perdana", 
    id: "2702268725", 
    role: "AI Engineer", 
    image: "/gading.jpg" 
  },
  { 
    name: "Jesslyn Trixie Edvilie", 
    id: "2702260514", 
    role: "Researcher", 
    image: "/jesslyn.jpg" 
  },
  { 
    name: "Owen Limantoro", 
    id: "2702262330", 
    role: "Researcher", 
    image: "/owen.jpg" 
  },
];

// --- UI COMPONENTS ---

const Button = ({ children, onClick, variant = 'primary', className = '', disabled = false, type = 'button', size = 'md' }) => {
  const baseStyle = "rounded-xl font-medium transition-all duration-300 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed active:scale-95 relative overflow-hidden group";
  
  const sizes = { sm: "px-3 py-1.5 text-xs", md: "px-5 py-2.5 text-sm", lg: "px-8 py-4 text-base" };
  
  const variants = {
    primary: "bg-indigo-600 text-white hover:bg-indigo-500 shadow-[0_0_20px_rgba(79,70,229,0.3)] hover:shadow-[0_0_30px_rgba(79,70,229,0.5)] border border-indigo-500/50",
    secondary: "bg-zinc-800 text-zinc-200 border border-zinc-700 hover:bg-zinc-700",
    ghost: "text-zinc-400 hover:text-white hover:bg-white/5",
    danger: "bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/20",
    outline: "border border-zinc-600 text-zinc-300 hover:border-indigo-400 hover:text-indigo-400 bg-transparent"
  };

  return (
    <button type={type} onClick={onClick} disabled={disabled} className={cn(baseStyle, sizes[size], variants[variant], className)}>
      <span className="relative z-10 flex items-center gap-2">{children}</span>
      {variant === 'primary' && <div className="absolute inset-0 bg-gradient-to-r from-indigo-600 to-violet-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />}
    </button>
  );
};

const Input = ({ label, type = "text", value, onChange, placeholder, required = false, icon: Icon, readOnly = false }) => (
  <div className="mb-5 group">
    <label className="block text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2 ml-1 group-focus-within:text-indigo-400 transition-colors">{label}</label>
    <div className="relative">
      {Icon && (
        <div className="absolute left-4 top-1/2 -translate-y-1/2 text-zinc-500 group-focus-within:text-indigo-400 transition-colors">
          <Icon size={18} />
        </div>
      )}
      <input
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required={required}
        readOnly={readOnly}
        className={cn(
          "w-full bg-zinc-900/50 border border-zinc-800 rounded-xl",
          "px-4 py-3 outline-none transition-all duration-300",
          "focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500/50",
          "text-white placeholder:text-zinc-600",
          readOnly && "opacity-60 cursor-not-allowed bg-zinc-900",
          Icon && "pl-11"
        )}
      />
    </div>
  </div>
);

const Card = ({ children, className = '', delay = 0 }) => (
  <motion.div 
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.5, delay }}
    className={cn(
      "bg-zinc-900/40 backdrop-blur-xl rounded-2xl border border-white/5",
      "shadow-xl shadow-black/20 hover:border-white/10 transition-colors",
      className
    )}
  >
    {children}
  </motion.div>
);

const Badge = ({ children, color = 'gray' }) => {
  const colors = {
    green: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20 shadow-[0_0_10px_rgba(16,185,129,0.1)]',
    red: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
    blue: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20 shadow-[0_0_10px_rgba(6,182,212,0.1)]',
    gray: 'bg-zinc-800 text-zinc-400 border-zinc-700',
    indigo: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20 shadow-[0_0_10px_rgba(99,102,241,0.1)]'
  };
  return (
    <span className={cn("px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider border", colors[color])}>
      {children}
    </span>
  );
};

const ResultCard = ({ name, res }) => (
  <div className="bg-zinc-900/80 backdrop-blur-md rounded-xl p-5 border border-white/10 hover:border-indigo-500/50 hover:bg-zinc-800/80 transition-all duration-300 group relative overflow-hidden flex flex-col h-full shadow-lg">
    {/* Decorative Gradient Background */}
    <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
    
    <div className="relative z-10 w-full flex flex-col h-full justify-between gap-4">
        {/* --- Header Section --- */}
        <div>
            <div className="flex justify-between items-start gap-3">
                <div className="flex items-start gap-3 min-w-0 flex-1">
                    {/* Icon */}
                    <div className={cn("mt-1 p-2 rounded-lg transition-all duration-300 shrink-0", 
                        res.has_object 
                            ? "bg-emerald-500/10 text-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.15)]" 
                            : "bg-zinc-800 text-zinc-500 group-hover:bg-zinc-700"
                    )}>
                        {name.includes("AKTP") ? <Brain size={18} /> : <Shield size={18} />}
                    </div>
                    
                    {/* Text Container */}
                    <div className="min-w-0 flex-1">
                        <h4 className="font-bold text-white text-sm leading-snug break-words">
                            {name}
                        </h4>
                        <div className="flex items-center gap-2 mt-1.5">
                            <span className="text-[10px] font-mono text-zinc-500 uppercase tracking-wider bg-black/40 px-1.5 py-0.5 rounded flex items-center gap-1">
                                <Clock size={10} /> {res.inference_time_ms.toFixed(0)}ms
                            </span>
                        </div>
                    </div>
                </div>
                
                {/* Status Badge - Fixed height/width to prevent shifting */}
                <div className="shrink-0 ml-1">
                    {res.has_object ? (
                        <div className="px-2 py-1 rounded-md text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 whitespace-nowrap shadow-sm">
                            DETECTED
                        </div>
                    ) : (
                        <div className="px-2 py-1 rounded-md text-[10px] font-bold bg-zinc-800 text-zinc-500 border border-zinc-700 whitespace-nowrap">
                            NONE
                        </div>
                    )}
                </div>
            </div>
        </div>
        
        {/* --- Bottom Section (Pushed to bottom via justify-between) --- */}
        <div>
            {/* Confidence Bar */}
            <div className="mb-4">
                <div className="flex justify-between items-baseline mb-2">
                    <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">Confidence</span>
                    <span className={cn("text-xl font-bold font-mono tracking-tight", res.has_object ? "text-emerald-400" : "text-zinc-500")}>
                        {(res.confidence * 100).toFixed(1)}%
                    </span>
                </div>
                <div className="w-full bg-zinc-950 rounded-full h-2 overflow-hidden border border-white/5 relative">
                    <motion.div 
                        initial={{ width: 0 }} 
                        animate={{ width: `${res.confidence * 100}%` }}
                        transition={{ duration: 1.5, ease: "easeOut" }}
                        className={cn(
                        "h-full rounded-full shadow-[0_0_10px_rgba(0,0,0,0.3)] relative", 
                        res.has_object ? "bg-emerald-500" : "bg-zinc-600"
                        )}
                    >
                         {/* Shine effect on bar */}
                         <div className="absolute top-0 right-0 bottom-0 w-[1px] bg-white/50 opacity-50 shadow-[0_0_5px_white]" />
                    </motion.div>
                </div>
            </div>
            
            {/* Stats Footer */}
            <div className="grid grid-cols-2 gap-2 border-t border-white/5 pt-3">
                {Object.entries(res.probabilities).slice(0, 2).map(([label, prob]) => (
                <div key={label} className="bg-black/40 rounded-lg py-2 px-3 flex flex-col items-center justify-center border border-white/5">
                    <span className="text-[9px] text-zinc-500 uppercase font-bold tracking-widest truncate max-w-full">{label}</span>
                    <span className="text-sm font-bold text-zinc-300 mt-0.5">{(prob * 100).toFixed(0)}%</span>
                </div>
                ))}
            </div>
        </div>
    </div>
  </div>
);

// --- LANDING PAGE ---

const LandingPage = ({ onStart, onLogin }) => {
  const [canViewPdf, setCanViewPdf] = useState(true);
  const [isVideoOpen, setIsVideoOpen] = useState(false);

  useEffect(() => {
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    setCanViewPdf(!isMobile);
  }, []);

  const handlePresentationClick = () => {
    const pdfUrl = "/presentation.pdf"; 
    if (canViewPdf) {
        window.open(pdfUrl, '_blank');
    } else {
        const link = document.createElement('a');
        link.href = pdfUrl;
        link.download = "AKTP_Presentation.pdf";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
  };

  return (
  <div className="min-h-screen bg-[#050505] text-white selection:bg-indigo-500/30 overflow-x-hidden font-sans">
    
    {/* Video Modal */}
    <AnimatePresence>
        {isVideoOpen && (
            <div 
                className="fixed inset-0 z-[100] flex items-center justify-center p-4 md:p-8 bg-black/90 backdrop-blur-md"
                onClick={() => setIsVideoOpen(false)}
            >
                <motion.div 
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    transition={{ type: "spring", damping: 25, stiffness: 300 }}
                    className="relative w-full max-w-6xl aspect-video bg-black rounded-2xl overflow-hidden shadow-2xl border border-white/10 flex items-center justify-center group"
                    onClick={(e) => e.stopPropagation()}
                >
                    <button 
                        onClick={() => setIsVideoOpen(false)} 
                        className="absolute top-4 right-4 z-10 p-2 bg-black/60 text-white rounded-full hover:bg-black/80 hover:scale-110 transition-all backdrop-blur-sm border border-white/10"
                    >
                        <X size={24}/>
                    </button>
                    <video 
                        src="/video.mp4" 
                        className="w-full h-full object-contain" 
                        controls 
                        autoPlay
                    >
                        Your browser does not support the video tag.
                    </video>
                </motion.div>
            </div>
        )}
    </AnimatePresence>

    <div className="fixed inset-0 z-0">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-indigo-800/20 rounded-full blur-[120px] animate-pulse" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-cyan-800/10 rounded-full blur-[120px]" />
        <div className="absolute top-[20%] right-[10%] w-[300px] h-[300px] bg-violet-800/10 rounded-full blur-[100px]" />
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:50px_50px] [mask-image:radial-gradient(ellipse_at_center,black_40%,transparent_100%)]" />
    </div>

    <nav className="fixed top-0 inset-x-0 z-50 bg-black/50 backdrop-blur-xl border-b border-white/5">
      <div className="max-w-7xl mx-auto px-4 md:px-6 h-16 md:h-20 flex justify-between items-center">
        <div className="font-bold text-xl flex items-center gap-3">
          <div className="w-8 h-8 md:w-10 md:h-10 rounded-lg md:rounded-xl flex items-center justify-center">
           <img src='/logo.png' alt="Logo"></img>
          </div>
          <span className="font-display tracking-tight text-white">ObjectDet</span>
        </div>
        <div className="flex gap-3">
          <Button variant="ghost" onClick={onLogin} size="sm" className="hidden sm:flex">Log In</Button>
          <Button onClick={onStart} size="sm" className="bg-white text-black hover:bg-zinc-200 border-none shadow-[0_0_20px_rgba(255,255,255,0.2)]">
            Get Started <ArrowRight size={16}/>
          </Button>
        </div>
      </div>
    </nav>

    <main className="relative z-10 pt-28 md:pt-48 pb-20 px-4 md:px-6 text-center">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs font-bold uppercase tracking-wider mb-8 backdrop-blur-sm">
            <Sparkles size={12} className="text-indigo-400" /> AKTP-SVM Model
          </div>
          
          <h1 className="text-4xl md:text-7xl font-bold tracking-tight mb-6 md:mb-8 leading-tight">
            See the Unseen with <br/>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-cyan-400 to-indigo-400">Adaptive AI Vision</span>
          </h1>
          
          <p className="text-base md:text-xl text-zinc-400 max-w-2xl mx-auto mb-10 leading-relaxed px-4">
            A breakthrough in lightweight object detection. Experience <span className="text-zinc-200 font-semibold">100% accuracy</span> with minimal latency, powered by student-teacher distillation into SVM.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center px-4">
            <Button onClick={onStart} size="lg" className="w-full sm:w-auto min-w-[200px]">
              Launch App
            </Button>
            <Button onClick={onLogin} variant="outline" size="lg" className="w-full sm:w-auto min-w-[200px]">
              View Specs
            </Button>
          </div>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, y: 40 }} 
          animate={{ opacity: 1, y: 0 }} 
          transition={{ delay: 0.3, duration: 0.8 }}
          className="mt-10 md:mt-20 max-w-6xl mx-auto px-2 md:px-0 grid grid-cols-1 md:grid-cols-2 gap-6"
        >
          {/* Card 1: Video/Image */}
          <div 
            onClick={() => setIsVideoOpen(true)}
            className="relative aspect-video rounded-2xl md:rounded-3xl bg-zinc-900 border border-white/10 shadow-2xl overflow-hidden group cursor-pointer hover:border-indigo-500/50 transition-colors"
          >
            {/* Live Video Preview (Muted/Autoplay) */}
            <video 
                src="/video.mp4" 
                className="absolute inset-0 w-full h-full object-cover opacity-60 group-hover:opacity-80 group-hover:scale-105 transition-all duration-700"
                muted
                loop
                playsInline
                autoPlay
            />
            
            <div className="absolute inset-0 bg-gradient-to-t from-black via-black/50 to-transparent pointer-events-none"></div>
            
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-16 h-16 md:w-20 md:h-20 bg-white/10 backdrop-blur-md rounded-full flex items-center justify-center border border-white/20 group-hover:bg-indigo-600 group-hover:border-indigo-500 transition-all duration-300 shadow-xl group-hover:scale-110">
                <Play className="fill-white text-white ml-1" size={24} />
              </div>
            </div>

            <div className="absolute bottom-6 left-6 text-left">
                <h3 className="text-white font-bold text-lg mb-1">Demo Footage</h3>
                <p className="text-zinc-400 text-xs">Watch AKTP in action</p>
            </div>
          </div>

          {/* Card 2: Presentation */}
          <div 
            onClick={handlePresentationClick}
            className="
                relative w-full rounded-2xl md:rounded-3xl bg-zinc-900 border border-white/10 shadow-2xl overflow-hidden group cursor-pointer 
                flex flex-col items-center justify-center p-8 text-center hover:border-indigo-500/50 transition-colors
                min-h-[320px] md:min-h-0 md:aspect-video
            "
          >
             <div className="w-16 h-16 bg-zinc-800 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform shadow-xl border border-white/5 shrink-0">
                <FileText size={32} className="text-indigo-400" />
             </div>
             <h3 className="text-xl font-bold text-white mb-2">Project Presentation</h3>
             <p className="text-zinc-400 text-sm mb-6 max-w-xs">View the full technical deck detailing our AKTP methodology and results.</p>
             <Button variant="secondary" size="sm" className="group-hover:bg-indigo-600 group-hover:text-white group-hover:border-indigo-500 pointer-events-none mt-auto md:mt-0">
                {canViewPdf ? <><Eye size={14} className="mr-1"/> View Deck</> : <><Download size={14} className="mr-1"/> Download PDF</>}
             </Button>
          </div>
        </motion.div>
    </main>
  </div>
  );
};
// --- AUTH COMPONENTS ---

const AuthLayout = ({ title, subtitle, children, error, success }) => (
  <div className="min-h-screen bg-black flex items-center justify-center p-4 relative overflow-hidden">
    <div className="absolute top-[-20%] left-[-10%] w-[600px] h-[600px] bg-indigo-900/20 rounded-full blur-[120px] pointer-events-none" />
    <div className="absolute bottom-[-20%] right-[-10%] w-[600px] h-[600px] bg-cyan-900/10 rounded-full blur-[120px] pointer-events-none" />
    
    <Card className="w-full max-w-md p-6 md:p-10 relative z-10 bg-zinc-900/80 border-white/10">
      <div className="text-center mb-10">
        <div className="w-16 h-16 mx-auto mb-6 relative flex items-center justify-center">
          <div className="absolute inset-0 rounded-full" />
          <img src='/logo.png' alt="Logo"></img>
        </div>
        <h1 className="text-3xl font-bold text-white mb-2">{title}</h1>
        <p className="text-zinc-500">{subtitle}</p>
      </div>
      
      {error && (
        <motion.div initial={{opacity:0, y:-10}} animate={{opacity:1, y:0}} className="mb-6 p-4 bg-red-500/10 border border-red-500/20 text-red-400 rounded-xl text-sm flex items-center gap-3">
            <AlertCircle size={18} className="shrink-0"/>{error}
        </motion.div>
      )}
      {success && (
        <motion.div initial={{opacity:0, y:-10}} animate={{opacity:1, y:0}} className="mb-6 p-4 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 rounded-xl text-sm flex items-center gap-3">
            <CheckCircle size={18} className="shrink-0"/>{success}
        </motion.div>
      )}
      
      {children}
    </Card>
  </div>
);

const AuthComponents = ({ 
  activeTab, setActiveTab, error, success, isLoading, 
  handleLogin, handleRegister, handleVerify, handleForgotPassword, handleResetPassword,
  authEmail, setAuthEmail, authPassword, setAuthPassword, 
  authName, setAuthName, authCode, setAuthCode,
  resetToken, setResetToken, newPassword, setNewPassword
}) => {
  if (activeTab === 'register') return (
    <AuthLayout title="Create ID" subtitle="Join the research network" error={error} success={success}>
      <form onSubmit={handleRegister} className="space-y-4">
        <Input label="Name" value={authName} onChange={e=>setAuthName(e.target.value)} icon={User} placeholder="John Doe"/>
        <Input label="Email" value={authEmail} onChange={e=>setAuthEmail(e.target.value)} icon={Brain} placeholder="name@example.com"/>
        <Input label="Password" type="password" value={authPassword} onChange={e=>setAuthPassword(e.target.value)} icon={Lock} placeholder="••••••••"/>
        <Button type="submit" className="w-full py-3 mt-4" disabled={isLoading}>{isLoading ? <Loader2 className="animate-spin"/> : 'Create Account'}</Button>
        <div className="text-center mt-6 space-y-3">
          <p className="text-sm text-zinc-500">
            Have an account? <button type="button" onClick={()=>setActiveTab('login')} className="text-indigo-400 hover:text-indigo-300 font-medium ml-1 transition-colors">Login</button>
          </p>
          <button type="button" onClick={()=>setActiveTab('landing')} className="text-xs text-zinc-600 hover:text-zinc-400 block w-full transition-colors">Return to Homepage</button>
        </div>
      </form>
    </AuthLayout>
  );

  if (activeTab === 'verify') return (
    <AuthLayout title="Verify Identity" subtitle="Enter the secure code sent to email" error={error} success={success}>
      <form onSubmit={handleVerify} className="space-y-4">
        <Input label="Security Code" value={authCode} onChange={e=>setAuthCode(e.target.value)} placeholder="000000" icon={Shield} />
        <Button type="submit" className="w-full py-3 mt-2" disabled={isLoading}>{isLoading ? <Loader2 className="animate-spin"/> : 'Verify Access'}</Button>
        <div className="text-center mt-6">
          <button type="button" onClick={()=>setActiveTab('login')} className="text-sm text-zinc-500 hover:text-zinc-300 transition-colors">Back to Login</button>
        </div>
      </form>
    </AuthLayout>
  );

  if (activeTab === 'forgot-password') return (
    <AuthLayout title="Reset Access" subtitle="We'll send a recovery token" error={error} success={success}>
      <form onSubmit={handleForgotPassword} className="space-y-4">
        <Input label="Email Address" value={authEmail} onChange={e=>setAuthEmail(e.target.value)} icon={Mail} placeholder="name@example.com" />
        <Button type="submit" className="w-full py-3 mt-2" disabled={isLoading}>{isLoading ? <Loader2 className="animate-spin"/> : 'Send Token'}</Button>
        <div className="flex flex-col gap-3 text-center mt-6">
          <button type="button" onClick={()=>setActiveTab('reset-password')} className="text-sm text-indigo-400 hover:text-indigo-300 font-medium">I already have a token</button>
          <button type="button" onClick={()=>setActiveTab('login')} className="flex items-center justify-center gap-2 text-sm text-zinc-500 hover:text-white transition-colors"><ArrowLeft size={16}/> Back</button>
        </div>
      </form>
    </AuthLayout>
  );

  if (activeTab === 'reset-password') return (
    <AuthLayout title="New Credentials" subtitle="Enter token and new password" error={error} success={success}>
      <form onSubmit={handleResetPassword} className="space-y-4">
        <Input label="Reset Token" value={resetToken} onChange={e=>setResetToken(e.target.value)} icon={Key} placeholder="Paste token here" />
        <Input label="New Password" type="password" value={newPassword} onChange={e=>setNewPassword(e.target.value)} icon={Lock} placeholder="New password" />
        <Button type="submit" className="w-full py-3 mt-2" disabled={isLoading}>{isLoading ? <Loader2 className="animate-spin"/> : 'Update Password'}</Button>
        <div className="text-center mt-6">
          <button type="button" onClick={()=>setActiveTab('login')} className="text-sm text-zinc-500 hover:text-white transition-colors">Back to Login</button>
        </div>
      </form>
    </AuthLayout>
  );

  return (
    <AuthLayout title="System Login" subtitle="Authenticate to access dashboard" error={error} success={success}>
      <form onSubmit={handleLogin} className="space-y-4">
        <Input label="Email" value={authEmail} onChange={e=>setAuthEmail(e.target.value)} icon={User} placeholder="name@example.com"/>
        <Input label="Password" type="password" value={authPassword} onChange={e=>setAuthPassword(e.target.value)} icon={Lock} placeholder="••••••••"/>
        <div className="flex justify-between items-center text-xs px-1">
          <button type="button" onClick={()=>setActiveTab('verify')} className="text-zinc-500 hover:text-white transition-colors">Verify Account</button>
          <button type="button" onClick={()=>setActiveTab('forgot-password')} className="text-indigo-400 hover:text-indigo-300 transition-colors">Forgot password?</button>
        </div>
        <Button type="submit" className="w-full py-3 mt-2" disabled={isLoading}>{isLoading ? <Loader2 className="animate-spin"/> : 'Authenticate'}</Button>
        <div className="text-center mt-6 space-y-3">
          <p className="text-sm text-zinc-500">
            No credentials? <button type="button" onClick={()=>setActiveTab('register')} className="text-indigo-400 hover:text-indigo-300 font-medium ml-1 transition-colors">Register</button>
          </p>
          <button type="button" onClick={()=>setActiveTab('landing')} className="text-xs text-zinc-600 hover:text-zinc-400 block w-full transition-colors">Return to Homepage</button>
        </div>
      </form>
    </AuthLayout>
  );
};

// --- VIEW COMPONENTS ---

const SpecsView = () => (
  <div className="max-w-4xl mx-auto space-y-6">
    <Card className="p-6 md:p-8">
      <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
        <FileText className="text-indigo-400" /> System Specifications
      </h2>
      <div className="grid md:grid-cols-2 gap-x-12 gap-y-6">
        {MODEL_SPECS.map((spec, i) => (
          <div key={i} className="flex flex-col sm:flex-row sm:justify-between border-b border-white/5 pb-3">
            <span className="text-zinc-500 font-medium text-sm">{spec.label}</span>
            <span className="text-zinc-200 font-semibold text-right text-sm">{spec.value}</span>
          </div>
        ))}
      </div>
    </Card>
    <div className="grid md:grid-cols-2 gap-6">
      <Card className="p-6 md:p-8">
        <h3 className="font-semibold mb-6 text-white text-lg">Distillation Pipeline</h3>
        <div className="space-y-6 relative">
          <div className="absolute left-[11px] top-2 bottom-2 w-[2px] bg-zinc-800" />
          {[
            'Image Preprocessing (224x224)', 
            'Train Teachers (ResNet + EffNet)', 
            'Compute Soft Labels & Entropy', 
            'Harris-SIFT Feature Extraction', 
            'Train Student SVM with AKTP Loss'
          ].map((step, i) => (
            <div key={i} className="flex items-center gap-4 relative z-10">
              <div className="w-6 h-6 rounded-full bg-zinc-900 border border-indigo-500/30 text-indigo-400 flex items-center justify-center text-[10px] font-bold shadow-[0_0_10px_rgba(99,102,241,0.2)]">{i+1}</div>
              <span className="text-sm text-zinc-300">{step}</span>
            </div>
          ))}
        </div>
      </Card>
      <Card className="p-6 md:p-8 flex flex-col justify-center items-center text-center bg-gradient-to-br from-zinc-900 to-indigo-900/10 border-indigo-500/10">
        <div className="w-16 h-16 bg-indigo-500/10 text-indigo-400 rounded-full flex items-center justify-center mb-4 ring-1 ring-indigo-500/30 animate-pulse"><Zap size={32}/></div>
        <h3 className="text-lg font-bold text-white">Student SVM</h3>
        <p className="text-sm text-zinc-400 mt-2 max-w-xs leading-relaxed">
          The student is a Support Vector Machine (SVM) trained to mimic the complex decision boundaries of deep learning teachers via Adaptive Knowledge Transfer.
        </p>
      </Card>
    </div>
  </div>
);

const TeamsView = () => (
  <div className="max-w-6xl mx-auto">
    <div className="mb-10 text-center">
      <h2 className="text-3xl font-bold text-white mb-3">Research Team</h2>
      <p className="text-zinc-400 max-w-2xl mx-auto text-sm">
        The brilliant minds behind the Object Detection AKTP project.
      </p>
    </div>
    
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
      {TEAM_MEMBERS.map((member, idx) => (
        <Card key={idx} className="p-0 border-0 group overflow-hidden bg-zinc-900/50 hover:bg-zinc-800 transition-all duration-300" delay={idx * 0.1}>
          <div className="h-1 bg-gradient-to-r from-indigo-500 to-purple-500 w-full"></div>
          
          <div className="p-6 flex flex-col items-center text-center relative z-10">
            <div className="p-1 rounded-full mb-4 shadow-xl border border-white/10 group-hover:border-indigo-500/50 transition-colors">
              <img 
                src={member.image} 
                alt={member.name} 
                className="w-20 h-20 rounded-full bg-zinc-800 object-cover"
              />
            </div>
            
            <h3 className="text-base font-bold text-white mb-1 group-hover:text-indigo-400 transition-colors">{member.name}</h3>
            <p className="text-xs font-medium text-zinc-500 mb-3 uppercase tracking-wider">{member.role}</p>
            
            <div className="w-full border-t border-white/5 my-3"></div>
            
            <div className="flex items-center gap-2">
                <span className="text-[10px] text-zinc-600 font-mono">NIM: {member.id}</span>
            </div>
          </div>
        </Card>
      ))}
    </div>
  </div>
);

const AnalyticsView = () => (
  <div className="space-y-6 animate-in fade-in duration-500">
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <Card className="p-5 border-l-4 border-l-blue-500">
        <div className="flex justify-between items-start">
            <div>
                <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-1">Total Images</p>
                <p className="text-2xl md:text-3xl font-bold text-white font-mono">101</p>
            </div>
            <Database className="text-blue-500 opacity-50" size={24} />
        </div>
      </Card>
      <Card className="p-5 border-l-4 border-l-emerald-500">
         <div className="flex justify-between items-start">
            <div>
                <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-1">Best Accuracy</p>
                <p className="text-2xl md:text-3xl font-bold text-white font-mono">100%</p>
                <p className="text-[10px] text-zinc-500 mt-1">AKTP-SVM</p>
            </div>
            <Target className="text-emerald-500 opacity-50" size={24} />
        </div>
      </Card>
      <Card className="p-5 border-l-4 border-l-indigo-500">
        <div className="flex justify-between items-start">
            <div>
                <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-1">Class Balance</p>
                <p className="text-2xl md:text-3xl font-bold text-white font-mono">3.39:1</p>
                <p className="text-[10px] text-zinc-500 mt-1">Pos : Neg</p>
            </div>
            <Users className="text-indigo-500 opacity-50" size={24} />
        </div>
      </Card>
       <Card className="p-5 border-l-4 border-l-orange-500">
        <div className="flex justify-between items-start">
            <div>
                <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-1">Categories</p>
                <p className="text-2xl md:text-3xl font-bold text-white font-mono">4</p>
            </div>
            <Layers className="text-orange-500 opacity-50" size={24} />
        </div>
      </Card>
    </div>

    {/* Performance Bar Chart */}
    <Card className="p-6 md:p-8">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
            <div>
                <h3 className="font-semibold text-white text-lg">Model Performance Comparison</h3>
                <div className="flex items-center gap-2 mt-1">
                    <p className="text-xs text-zinc-500">Vanilla SVM vs Distilled SVM vs AKTP-SVM</p>
                    <span className="hidden md:inline text-zinc-700">•</span>
                    <Badge color="blue">Binary Object Detection</Badge>
                </div>
            </div>
            <div className="flex gap-4 text-xs flex-wrap">
                <div className="flex items-center gap-1"><div className="w-2 h-2 bg-slate-500 rounded-full"></div> Vanilla</div>
                <div className="flex items-center gap-1"><div className="w-2 h-2 bg-orange-500 rounded-full"></div> Distilled</div>
                <div className="flex items-center gap-1"><div className="w-2 h-2 bg-emerald-500 rounded-full"></div> AKTP</div>
            </div>
        </div>
        <div className="h-[300px] md:h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart 
                data={MODEL_PERFORMANCE} 
                margin={{top: 20, right: 10, left: -20, bottom: 0}}
                barGap={2}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" opacity={0.5} vertical={false}/>
              <XAxis 
                dataKey="name" 
                axisLine={false} 
                tickLine={false} 
                tick={{fill: '#a1a1aa', fontSize: 11, fontWeight: 500}} 
                dy={10}
                interval={0} 
              />
              <YAxis 
                domain={[0.8, 1.05]} 
                stroke="#71717a" 
                tick={{fill: '#71717a', fontSize: 11}} 
                tickLine={false} 
                axisLine={false} 
              />
              <Tooltip 
                contentStyle={{backgroundColor: '#18181b', border: '1px solid #27272a', borderRadius: '8px', color:'#f4f4f5'}} 
                itemStyle={{color: '#e4e4e7', fontSize: '12px'}} 
                cursor={{fill: 'rgba(255,255,255,0.05)'}}
              />
              <Bar dataKey="Accuracy" fill="#64748b" radius={[4, 4, 0, 0]} name="Accuracy" />
              <Bar dataKey="Precision" fill="#f97316" radius={[4, 4, 0, 0]} name="Precision" />
              <Bar dataKey="Recall" fill="#10b981" radius={[4, 4, 0, 0]} name="Recall" />
              <Bar dataKey="F1" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="F1 Score" />
            </BarChart>
          </ResponsiveContainer>
        </div>
    </Card>

    <div className="grid lg:grid-cols-2 gap-6">
       {/* Category Distribution Pie Chart */}
       <Card className="p-6">
            <h3 className="font-semibold text-white mb-6">Category Distribution</h3>
            <div className="flex flex-col sm:flex-row items-center justify-center">
                <div className="h-64 w-64 relative">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={CLASS_DISTRIBUTION}
                                cx="50%"
                                cy="50%"
                                innerRadius={60}
                                outerRadius={80}
                                paddingAngle={5}
                                dataKey="value"
                            >
                                {CLASS_DISTRIBUTION.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.color} stroke="rgba(0,0,0,0)" />
                                ))}
                            </Pie>
                            <Tooltip contentStyle={{backgroundColor: '#18181b', border: '1px solid #27272a', borderRadius: '8px'}}/>
                        </PieChart>
                    </ResponsiveContainer>
                    <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                        <span className="text-3xl font-bold text-white">101</span>
                        <span className="text-xs text-zinc-500">Images</span>
                    </div>
                </div>
                <div className="mt-6 sm:mt-0 sm:ml-8 space-y-3 w-full max-w-xs">
                    {CLASS_DISTRIBUTION.map((item) => (
                        <div key={item.name} className="flex items-center justify-between">
                            <div className="flex items-center">
                                <span className="w-3 h-3 rounded-full mr-3" style={{ backgroundColor: item.color }}></span>
                                <span className="text-sm text-zinc-300">{item.name}</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="text-sm font-bold text-white">{item.value}</span>
                                <span className="text-xs text-zinc-500">({((item.value / 101) * 100).toFixed(1)}%)</span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
       </Card>

       {/* Binary Classification Split */}
       <Card className="p-6">
            <h3 className="font-semibold text-white mb-6">Binary Classification Split</h3>
            
            <div className="space-y-8">
                <div>
                    <div className="flex justify-between items-end mb-2">
                        <span className="text-sm text-emerald-400 font-medium flex items-center gap-2">
                            <CheckCircle size={14}/> Positive (Objects)
                        </span>
                        <span className="text-2xl font-bold text-white">78 <span className="text-sm text-zinc-500 font-normal">images</span></span>
                    </div>
                    <div className="w-full bg-zinc-800 rounded-full h-2 overflow-hidden mb-3">
                         <div className="h-full bg-emerald-500 rounded-full" style={{ width: '77.2%' }}></div>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                        <div className="bg-zinc-800/50 rounded p-2 text-center border border-white/5">
                            <div className="text-xs text-zinc-500">Normal</div>
                            <div className="font-mono font-bold text-emerald-400">23</div>
                        </div>
                        <div className="bg-zinc-800/50 rounded p-2 text-center border border-white/5">
                            <div className="text-xs text-zinc-500">Dark</div>
                            <div className="font-mono font-bold text-emerald-400">26</div>
                        </div>
                        <div className="bg-zinc-800/50 rounded p-2 text-center border border-white/5">
                            <div className="text-xs text-zinc-500">Angle</div>
                            <div className="font-mono font-bold text-emerald-400">29</div>
                        </div>
                    </div>
                </div>

                <div>
                     <div className="flex justify-between items-end mb-2">
                        <span className="text-sm text-red-400 font-medium flex items-center gap-2">
                            <AlertCircle size={14}/> Negative (Empty)
                        </span>
                        <span className="text-2xl font-bold text-white">23 <span className="text-sm text-zinc-500 font-normal">images</span></span>
                    </div>
                    <div className="w-full bg-zinc-800 rounded-full h-2 overflow-hidden mb-2">
                         <div className="h-full bg-red-500 rounded-full" style={{ width: '22.8%' }}></div>
                    </div>
                    <p className="text-xs text-zinc-500">Background images with no stuffed animals present.</p>
                </div>
            </div>
       </Card>
    </div>
  </div>
);

// --- HISTORY COMPONENTS ---

const HistoryItem = ({ rec }) => {
  const [expanded, setExpanded] = useState(false);
  const [showAnnotated, setShowAnnotated] = useState(false);
  
  const mainResult = rec.results?.['AKTP-SVM'];
  const hasAnnotation = !!rec.annotated_image_url;
  
  useEffect(() => {
    if (!expanded) setShowAnnotated(false);
  }, [expanded]);

  const displayImage = showAnnotated && hasAnnotation ? rec.annotated_image_url : rec.image_url;

  return (
    <div className="border-b border-white/5 last:border-0 bg-zinc-900/20">
      {/* Header Row */}
      <div 
        onClick={() => setExpanded(!expanded)}
        className="p-4 flex flex-col sm:flex-row sm:items-center gap-4 hover:bg-white/5 transition-colors cursor-pointer group"
      >
        <div className="flex items-center gap-4 w-full sm:w-auto">
            <div className="w-14 h-14 bg-zinc-800 rounded-lg overflow-hidden flex-shrink-0 border border-white/10 relative">
                <img src={rec.image_url} className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" alt="" />
                {hasAnnotation && (
                    <div className="absolute top-1 right-1 w-2 h-2 bg-emerald-500 rounded-full shadow-[0_0_5px_rgba(16,185,129,0.5)]"></div>
                )}
            </div>
            
            <div className="flex-1 min-w-0 sm:hidden">
                <div className="flex items-center justify-between">
                    <span className={cn("font-bold text-sm", mainResult?.has_object ? "text-emerald-400" : "text-zinc-300")}>
                        {mainResult?.label || 'Unknown'}
                    </span>
                    <span className="text-xs font-bold text-indigo-400">{(mainResult?.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="text-[10px] text-zinc-500 mt-1 flex gap-2">
                    <span>{new Date(rec.created_at).toLocaleDateString()}</span>
                    <span>{rec._id.slice(-6)}</span>
                </div>
            </div>
        </div>
        
        <div className="hidden sm:block flex-1 min-w-0">
          <div className="flex items-center gap-3">
            <span className={cn("font-bold text-base", mainResult?.has_object ? "text-emerald-400" : "text-zinc-300")}>
              {mainResult?.label || 'Unknown'}
            </span>
            <Badge color="gray">{rec._id.slice(-8)}</Badge>
            {hasAnnotation && <Badge color="indigo">Boxes</Badge>}
          </div>
          <div className="flex gap-4 mt-1 text-xs text-zinc-500">
             <span className="flex items-center gap-1"><Calendar size={12} /> {new Date(rec.created_at).toLocaleDateString()}</span>
             <span className="flex items-center gap-1"><Clock size={12} /> {new Date(rec.created_at).toLocaleTimeString()}</span>
          </div>
        </div>

        <div className="hidden sm:flex items-center gap-6 mr-2">
          <div className="text-right">
            <div className="text-[10px] uppercase text-zinc-500 font-bold tracking-wider">Confidence</div>
            <div className="text-lg font-bold text-indigo-400 font-mono">{(mainResult?.confidence * 100).toFixed(1)}%</div>
          </div>
          {expanded ? <ChevronUp className="text-zinc-500" size={20} /> : <ChevronDown className="text-zinc-500 group-hover:text-white" size={20} />}
        </div>
        
        <div className="sm:hidden w-full flex justify-center pt-2">
            {expanded ? <ChevronUp className="text-zinc-600" size={16} /> : <ChevronDown className="text-zinc-600" size={16} />}
        </div>
      </div>

      <AnimatePresence>
        {expanded && (
          <motion.div 
            initial={{ height: 0, opacity: 0 }} 
            animate={{ height: 'auto', opacity: 1 }} 
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden bg-black/40 shadow-inner"
          >
            <div className="p-4 md:p-6 border-t border-white/5">
                <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
                    {/* Image Viewer */}
                    <div className="xl:col-span-3 flex flex-col gap-3">
                        <div className="relative rounded-xl overflow-hidden bg-zinc-950 border border-white/10 aspect-square xl:aspect-[3/4] group/img shadow-2xl">
                            <img 
                                src={displayImage} 
                                alt="Result" 
                                className="w-full h-full object-contain p-2"
                            />
                            {/* View Toggle Overlay */}
                            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 flex gap-1 p-1 bg-zinc-900/90 backdrop-blur-md rounded-lg border border-white/10 opacity-0 group-hover/img:opacity-100 transition-opacity shadow-xl">
                                 <button 
                                    onClick={() => setShowAnnotated(false)}
                                    className={cn("p-2 rounded transition-colors", !showAnnotated ? "bg-white/20 text-white" : "text-zinc-400 hover:text-white")}
                                    title="Original Image"
                                 >
                                    <ImageIcon size={16} />
                                 </button>
                                 {hasAnnotation && (
                                    <button 
                                        onClick={() => setShowAnnotated(true)}
                                        className={cn("p-2 rounded transition-colors", showAnnotated ? "bg-indigo-500 text-white" : "text-zinc-400 hover:text-white")}
                                        title="Annotated Image"
                                    >
                                        <ScanEye size={16} />
                                    </button>
                                 )}
                            </div>
                        </div>
                        
                        <div className="flex justify-between items-center px-2">
                            <p className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">
                                {showAnnotated ? "Annotated View" : "Original Source"}
                            </p>
                            <a 
                                href={displayImage} 
                                target="_blank" 
                                rel="noopener noreferrer" 
                                className="text-[10px] text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-bold uppercase tracking-wide"
                            >
                                <Download size={12} /> Save Image
                            </a>
                        </div>
                    </div>

                    {/* Results Grid */}
                    <div className="xl:col-span-9">
                         <div className="flex items-center justify-between mb-4">
                            <h3 className="text-sm font-bold text-zinc-400 uppercase tracking-widest flex items-center gap-2">
                                <Layers size={14} /> Model Ensemble Results
                            </h3>
                         </div>
                         
                         {/* FIX: Increased width to 300px min. 
                             This forces cards to be wider, preventing awkward text breaks. */}
                         <div className="grid grid-cols-1 md:grid-cols-2 2xl:grid-cols-[repeat(auto-fill,minmax(300px,1fr))] gap-4">
                            {Object.entries(rec.results || {}).map(([name, res]) => (
                                <motion.div 
                                    key={name} 
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    whileInView={{ opacity: 1, scale: 1 }}
                                    transition={{ duration: 0.3 }}
                                    className="h-full"
                                > 
                                    <ResultCard name={name} res={res} />
                                </motion.div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const HistoryView = ({ history, fetchHistory }) => {
  useEffect(() => { fetchHistory() }, []); 

  return (
    <Card className="overflow-hidden">
      <div className="p-5 md:p-6 border-b border-white/5 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 bg-zinc-900/50">
        <div>
          <h3 className="font-semibold text-white text-lg">Detection Archives</h3>
          <p className="text-sm text-zinc-500">Historical analysis data</p>
        </div>
        <Button variant="secondary" size="sm" onClick={fetchHistory}><History size={16}/> Refresh Data</Button>
      </div>
      <div className="bg-transparent">
        {history.length > 0 ? history.map((rec) => (
          <HistoryItem key={rec._id} rec={rec} />
        )) : (
          <div className="p-16 text-center text-zinc-600">
            <History size={48} className="mx-auto mb-4 opacity-20" />
            <p>No history records found.</p>
          </div>
        )}
      </div>
    </Card>
  );
};

const ProfileView = ({ userProfile, setUserProfile, onAvatarUpload, onChangePassword, isLoading }) => {
  const [oldPassword, setOldPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [msg, setMsg] = useState({ text: '', type: '' });
  const fileInputRef = useRef(null);

  const handleSave = async (e) => {
    e.preventDefault();
    setMsg({ text: '', type: '' });
    
    if (newPassword !== confirmPassword) {
      setMsg({ text: "Passwords do not match.", type: 'error' });
      return;
    }
    
    if (newPassword.length < 8) {
        setMsg({ text: "Password must be at least 8 characters.", type: 'error' });
        return;
    }

    try {
      await onChangePassword(oldPassword, newPassword);
      setMsg({ text: "Password changed successfully!", type: 'success' });
      setOldPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (err) {
      setMsg({ text: err.message || "Failed to update password", type: 'error' });
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const error = validateFile(file);
    if (error) { alert(error); return; }
    onAvatarUpload(file);
  };

  return (
    <div className="max-w-2xl mx-auto">
      <Card className="p-6 md:p-10">
        <div className="flex flex-col items-center mb-10">
          <div className="relative group cursor-pointer" onClick={() => fileInputRef.current.click()}>
            <div className="w-28 h-28 rounded-full p-1 bg-gradient-to-tr from-indigo-500 to-cyan-500 shadow-2xl shadow-indigo-500/20">
              <img src={userProfile.avatar} alt="Profile" className="w-full h-full rounded-full bg-black object-cover border-4 border-black" />
            </div>
            <div className="absolute inset-0 flex items-center justify-center bg-black/60 rounded-full opacity-0 group-hover:opacity-100 transition-opacity backdrop-blur-sm">
              <Camera className="text-white" size={24} />
            </div>
            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              accept="image/png, image/jpeg, image/jpg" 
              onChange={handleFileChange} 
            />
          </div>
          <h2 className="mt-6 text-2xl font-bold text-white flex items-center gap-2">
            {userProfile.name}
          </h2>
          <div className="flex items-center gap-2 mt-2 px-4 py-1.5 bg-zinc-900 rounded-full border border-white/5">
            <Brain size={14} className="text-indigo-400" />
            <p className="text-sm text-zinc-300">{userProfile.email}</p>
          </div>
        </div>

        <form onSubmit={handleSave} className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            <Input label="Full Name" value={userProfile.name} readOnly icon={User} />
            <Input label="Email Address" value={userProfile.email} readOnly icon={Brain} />
          </div>
          
          <div className="border-t border-white/5 pt-8">
            <h3 className="text-xs font-bold mb-6 text-indigo-400 uppercase tracking-widest flex items-center gap-2">
              <Lock size={14} /> Security Settings
            </h3>
            <div className="space-y-4">
              <Input 
                label="Current Password" 
                type="password" 
                value={oldPassword} 
                onChange={(e) => setOldPassword(e.target.value)} 
                icon={Lock} 
                placeholder="Required for changes"
                required
              />
              <div className="grid md:grid-cols-2 gap-4">
                <Input 
                    label="New Password" 
                    type="password" 
                    value={newPassword} 
                    onChange={(e) => setNewPassword(e.target.value)} 
                    icon={Lock} 
                    placeholder="Min. 8 chars"
                    required
                />
                <Input 
                    label="Confirm New" 
                    type="password" 
                    value={confirmPassword} 
                    onChange={(e) => setConfirmPassword(e.target.value)} 
                    icon={Lock} 
                    placeholder="Repeat password"
                    required
                />
              </div>
            </div>
          </div>
          
          <div className="flex flex-col-reverse sm:flex-row justify-between items-center pt-4 gap-4">
            <span className={cn("text-sm font-medium", msg.type === 'error' ? 'text-red-400' : 'text-emerald-400')}>
              {msg.text}
            </span>
            <Button type="submit" variant="primary" disabled={isLoading} className="w-full sm:w-auto">
              {isLoading ? 'Updating...' : 'Update Credentials'}
            </Button>
          </div>
        </form>
      </Card>
    </div>
  );
};

const DashboardView = ({ 
  selectedFile, previewUrl, predictionResult, isLoading, 
  handleFileSelect, handlePredict, handleReset
}) => {
  const [showAnnotated, setShowAnnotated] = useState(false);

  useEffect(() => {
    if (predictionResult?.annotated_image_url) {
      setShowAnnotated(true);
    } else {
      setShowAnnotated(false);
    }
  }, [predictionResult]);

  const currentImage = (showAnnotated && predictionResult?.annotated_image_url) 
    ? predictionResult.annotated_image_url 
    : previewUrl;

  return (
    <div className="space-y-6 animate-in fade-in duration-500 pb-20 md:pb-0">
      <div className="flex flex-col xl:grid xl:grid-cols-12 gap-6 items-start">
        
        {/* --- INPUT SECTION --- */}
        <div className="w-full xl:col-span-5 2xl:col-span-4 xl:sticky xl:top-8 z-10">
            <Card className="p-0 overflow-hidden bg-zinc-900/50 border-zinc-800 shadow-xl">
                {/* Header */}
                <div className="p-4 border-b border-white/5 bg-zinc-900/80 flex justify-between items-center">
                    <div>
                        <h2 className="font-bold text-base text-white flex items-center gap-2">
                            <ScanLine className="text-indigo-400" size={18} /> Input Source
                        </h2>
                    </div>
                    {/* Status badge */}
                    {predictionResult && (
                         <div className="xl:hidden px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 text-[10px] font-bold border border-emerald-500/20">
                            COMPLETE
                         </div>
                    )}
                </div>
                
                <div className="p-4">
                    {/* Image Area */}
                    <div className="relative group w-full">
                        <div 
                            className={cn(
                            "border-2 border-dashed rounded-xl transition-all relative z-10 w-full overflow-hidden bg-black/40 flex items-center justify-center",
                            "aspect-square xl:aspect-[4/3]", // Square on mobile for visibility, landscape on desktop
                            previewUrl ? "border-indigo-500/30" : "border-zinc-700 hover:border-indigo-500 hover:bg-zinc-800/50"
                            )}
                        >
                            <input 
                                type="file" 
                                onChange={handleFileSelect} 
                                className={cn("absolute inset-0 w-full h-full opacity-0 z-20", !predictionResult && "cursor-pointer")}
                                accept="image/png, image/jpeg, image/jpg" 
                                disabled={!!predictionResult}
                            />
                            
                            {currentImage ? (
                                <div className="relative w-full h-full flex items-center justify-center bg-black">
                                    <img 
                                        src={currentImage} 
                                        alt="Preview" 
                                        className="w-full h-full object-contain" 
                                    />
                                    
                                    {showAnnotated && (
                                        <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(rgba(18,16,255,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_4px,3px_100%] animate-pulse" />
                                    )}

                                    {!predictionResult && (
                                        <div className="absolute inset-0 flex items-center justify-center bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity backdrop-blur-sm pointer-events-none">
                                            <p className="text-white font-medium flex items-center gap-2 text-sm"><ImageIcon size={16}/> Replace</p>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="text-center p-4">
                                    <div className="w-12 h-12 bg-zinc-800 rounded-xl flex items-center justify-center mx-auto mb-3 border border-white/5">
                                        <Upload size={20} className="text-indigo-400" />
                                    </div>
                                    <p className="text-sm font-bold text-white">Tap to Upload</p>
                                    <p className="text-[10px] text-zinc-500 mt-1 uppercase tracking-wider">JPG, PNG (Max 5MB)</p>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* View Toggle Controls */}
                    {predictionResult && predictionResult.annotated_image_url && (
                        <div className="mt-4 flex bg-zinc-950 rounded-lg p-1 border border-white/5">
                            <button 
                                onClick={() => setShowAnnotated(false)}
                                className={cn(
                                    "flex-1 py-2.5 text-xs font-bold rounded-md transition-all flex items-center justify-center gap-2 touch-manipulation",
                                    !showAnnotated ? "bg-zinc-800 text-white shadow-md" : "text-zinc-500"
                                )}
                            >
                                <ImageIcon size={14} /> Original
                            </button>
                            <button 
                                onClick={() => setShowAnnotated(true)}
                                className={cn(
                                    "flex-1 py-2.5 text-xs font-bold rounded-md transition-all flex items-center justify-center gap-2 touch-manipulation",
                                    showAnnotated ? "bg-indigo-600 text-white shadow-md shadow-indigo-500/20" : "text-zinc-500"
                                )}
                            >
                                <ScanEye size={14} /> AI Vision
                            </button>
                        </div>
                    )}
                    
                    {/* Action Buttons */}
                    <div className="mt-4">
                        {!predictionResult ? (
                            <Button 
                                onClick={handlePredict} 
                                className="w-full py-3.5 text-sm font-bold shadow-lg shadow-indigo-900/20 active:scale-[0.98]" 
                                disabled={!selectedFile || isLoading}
                            >
                                {isLoading ? <><Loader2 className="animate-spin" /> Processing...</> : 'Analyze Image'}
                            </Button>
                        ) : (
                            <Button 
                                onClick={() => { handleReset(); setShowAnnotated(false); }} 
                                variant="secondary" 
                                className="w-full py-3.5 text-sm font-bold border-zinc-700 bg-zinc-800/80 active:scale-[0.98]"
                            >
                                <Trash2 size={16} /> New Analysis
                            </Button>
                        )}
                    </div>
                </div>
            </Card>
        </div>

        {/* --- RESULTS SECTION --- */}
        <div className="w-full xl:col-span-7 2xl:col-span-8">
          {predictionResult ? (
            <div className="space-y-4">
                {/* Desktop Header */}
                <div className="hidden xl:flex justify-between items-end border-b border-white/10 pb-4 mb-6">
                    <div>
                        <h2 className="text-2xl font-bold text-white">Inference Results</h2>
                        <p className="text-zinc-500 text-sm">Ensemble model consensus</p>
                    </div>
                    <Badge color="green">Complete</Badge>
                </div>

                {/* Mobile Header */}
                <div className="xl:hidden flex items-center gap-2 mb-2 px-1">
                    <Layers size={16} className="text-indigo-400"/>
                    <h3 className="text-sm font-bold text-zinc-300 uppercase tracking-wider">Model Results</h3>
                </div>

                {/* --- FIX: RESPONSIVE LAYOUT --- */}
                {/* Mobile/Tablet (< xl): Horizontal Scroll (Carousel).
                    Desktop (>= xl): Grid with Auto-Fill. 
                    
                    'minmax(280px, 1fr)' prevents the cards from getting too narrow on desktop.
                    'min-w-[85vw]' ensures cards are wide enough on mobile.
                */}
                <div className="
                    flex overflow-x-auto pb-4 gap-3 snap-x snap-mandatory -mx-4 px-4 
                    xl:grid xl:grid-cols-[repeat(auto-fill,minmax(280px,1fr))] xl:overflow-visible xl:pb-0 xl:gap-4 xl:mx-0 xl:px-0
                    scrollbar-hide
                ">
                    {Object.entries(predictionResult.results || {}).map(([name, res], idx) => (
                        <div 
                            key={name} 
                            // Mobile width: 85% of screen. Desktop: auto based on grid.
                            className="min-w-[85vw] sm:min-w-[350px] xl:min-w-0 snap-center h-full"
                        >
                            <motion.div 
                                initial={{ opacity: 0, y: 10 }} 
                                animate={{ opacity: 1, y: 0 }} 
                                transition={{ delay: idx * 0.1 }}
                                className="h-full"
                            >
                                <ResultCard name={name} res={res} />
                            </motion.div>
                        </div>
                    ))}
                </div>
                
                {/* Mobile Scroll Hint */}
                <div className="xl:hidden text-center text-[10px] text-zinc-600 font-medium animate-pulse mt-2">
                    Swipe for more models &rarr;
                </div>
            </div>
          ) : (
            /* --- EMPTY STATE (Fixed: Visible on Mobile) --- */
            <Card className="h-full min-h-[300px] flex flex-col items-center justify-center p-8 bg-zinc-900/20 border-dashed border-zinc-800 text-center">
              <div className="w-16 h-16 md:w-20 md:h-20 bg-zinc-900 rounded-full flex items-center justify-center mb-6 text-zinc-700 shadow-inner border border-white/5 relative">
                 <div className="absolute inset-0 border border-indigo-500/10 rounded-full animate-ping opacity-20" />
                <BarChart3 size={32} className="opacity-50" />
              </div>
              <h3 className="text-lg font-bold text-white mb-2">System Ready</h3>
              <p className="text-zinc-500 max-w-xs text-sm leading-relaxed">
                Upload an image to start the AKTP-SVM ensemble analysis.
              </p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};
// --- ROOT APP ---

export default function App() {
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [activeTab, setActiveTab] = useState('dashboard');
  const [authView, setAuthView] = useState('landing');
  
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [userProfile, setUserProfile] = useState({ 
    name: "User", 
    email: localStorage.getItem('userEmail') || "", 
    avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=User" 
  });

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  
  // Auth State
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authName, setAuthName] = useState('');
  const [authCode, setAuthCode] = useState('');
  
  // Reset Password State
  const [resetToken, setResetToken] = useState('');
  const [newPassword, setNewPassword] = useState('');
  
  // Dashboard State
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  
  // History State
  const [history, setHistory] = useState([]);

  useEffect(() => {
    document.documentElement.classList.add('dark');
  }, []);

  useEffect(() => {
    if (token && ['landing', 'login', 'register', 'verify', 'forgot-password', 'reset-password'].includes(activeTab)) {
      setActiveTab('dashboard');
    }
  }, [token]);

  const apiCall = async (endpoint, method = 'GET', body = null, isFileUpload = false) => {
    setIsLoading(true); setError(''); setSuccess('');
    try {
      const headers = {};
      if (!isFileUpload) headers['Content-Type'] = 'application/json';
      if (token) headers['Authorization'] = `Bearer ${token}`;
      const response = await fetch(`${API_URL}${endpoint}`, {
        method, headers, body: isFileUpload ? body : (body ? JSON.stringify(body) : null)
      });

      if (response.status === 401) {
        setToken(null);
        localStorage.removeItem('token');
        localStorage.removeItem('userEmail');
        setAuthView('login');
        setError("Session expired. Please log in again.");
        throw new Error("Session expired. Please log in again.");
      }

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Something went wrong');
      return data;
    } catch (err) { 
        if(err.message !== "Session expired. Please log in again.") {
            setError(err.message); 
        }
        return null; 
    } finally { setIsLoading(false); }
  };

  useEffect(() => {
    if (token) {
      const fetchProfile = async () => {
        const data = await apiCall('/auth/me');
        if (data) {
          setUserProfile({
            name: data.full_name || "User",
            email: data.email,
            avatar: data.profile_picture || `https://api.dicebear.com/7.x/avataaars/svg?seed=${data.email}`
          });
        }
      };
      fetchProfile();
    }
  }, [token]);

  const handleAvatarUpload = async (file) => {
    const error = validateFile(file);
    if (error) { setError(error); return; }

    const formData = new FormData();
    formData.append('file', file);
    const data = await apiCall('/auth/me/avatar', 'POST', formData, true);
    if (data) {
      setUserProfile(prev => ({ ...prev, avatar: data.url }));
      setSuccess('Profile picture updated successfully!');
      setTimeout(() => setSuccess(''), 3000);
    }
  };
  
  const handleChangePassword = async (oldPassword, newPassword) => {
    const data = await apiCall('/auth/change-password', 'POST', { 
        old_password: oldPassword, 
        new_password: newPassword 
    });
    if (!data) throw new Error("Failed to change password");
    return data;
  };

  const handleLogin = async (e) => { 
    e.preventDefault(); 
    const data = await apiCall('/auth/login', 'POST', { email: authEmail, password: authPassword }); 
    if (data) { 
      setToken(data.access_token); 
      localStorage.setItem('token', data.access_token);
      localStorage.setItem('userEmail', authEmail); 
    }
  };

  const handleRegister = async (e) => { 
    e.preventDefault(); 
    const data = await apiCall('/auth/register', 'POST', { email: authEmail, password: authPassword, full_name: authName }); 
    if (data) { 
      localStorage.setItem('token', data.access_token);
      localStorage.setItem('userEmail', authEmail); 
      setUserProfile(p => ({...p, email: authEmail}));
      setAuthView('verify'); 
      setSuccess('Verification code sent to your email!'); 
    }
  };

  const handleVerify = async (e) => { 
    e.preventDefault(); 
    const data = await apiCall('/auth/verify', 'POST', { email: authEmail, code: authCode }); 
    if (data) { 
      const savedToken = localStorage.getItem('token');
      if (savedToken) {
        setToken(savedToken); 
        setActiveTab('dashboard'); 
        setSuccess('Account verified successfully!'); 
      } else {
        setAuthView('login');
        setSuccess('Verified! Please log in.');
      }
    }
  };

  const handleForgotPassword = async (e) => {
    e.preventDefault();
    const data = await apiCall('/auth/forgot-password', 'POST', { email: authEmail });
    if (data) {
      setSuccess(data.message);
    }
  };

  const handleResetPassword = async (e) => {
    e.preventDefault();
    const data = await apiCall('/auth/reset-password', 'POST', { token: resetToken, new_password: newPassword });
    if (data) {
      setAuthView('login');
      setSuccess('Password reset successful! Please login.');
    }
  };
  
  const handleFileSelect = (e) => {
    const f = e.target.files[0];
    if (f) { 
        const error = validateFile(f);
        if (error) { alert(error); return; }
        setSelectedFile(f); 
        setPreviewUrl(URL.createObjectURL(f)); 
        setPredictionResult(null); 
        e.target.value = null; 
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setPredictionResult(null);
  };

  const handlePredict = async () => { 
    if (!selectedFile) return; 
    const formData = new FormData(); 
    formData.append('file', selectedFile); 
    const data = await apiCall('/api/predict', 'POST', formData, true); 
    if (data) setPredictionResult(data); 
  };
  
  const fetchHistory = async () => { const data = await apiCall('/api/history'); if (data) setHistory(data); };

  if (!token) {
    if (authView === 'landing') {
        return (
            <LandingPage 
                onLogin={() => setAuthView('login')} 
                onStart={() => setAuthView('register')}
            />
        );
    }

    return (
      <AuthComponents 
        activeTab={authView} setActiveTab={setAuthView}
        error={error} success={success} isLoading={isLoading}
        handleLogin={handleLogin} handleRegister={handleRegister} 
        handleVerify={handleVerify} handleForgotPassword={handleForgotPassword}
        handleResetPassword={handleResetPassword}
        authEmail={authEmail} setAuthEmail={setAuthEmail}
        authPassword={authPassword} setAuthPassword={setAuthPassword}
        authName={authName} setAuthName={setAuthName}
        authCode={authCode} setAuthCode={setAuthCode}
        resetToken={resetToken} setResetToken={setResetToken}
        newPassword={newPassword} setNewPassword={setNewPassword}
      />
    );
  }

  const NavItem = ({ id, icon: I, label }) => (
    <button 
        onClick={()=>{setActiveTab(id);setIsMobileMenuOpen(false)}} 
        className={cn(
            "w-full flex items-center gap-4 px-4 py-3.5 rounded-xl transition-all duration-200 group relative overflow-hidden", 
            activeTab===id 
                ? "bg-gradient-to-r from-indigo-600 to-indigo-700 text-white shadow-lg shadow-indigo-900/20 border border-white/10" 
                : "text-zinc-400 hover:bg-white/5 hover:text-white"
        )}
    >
        <I size={20} className={cn("transition-colors", activeTab===id ? "text-white" : "text-zinc-500 group-hover:text-indigo-400")} />
        <span className="font-medium text-sm tracking-wide">{label}</span>
        {activeTab === id && <div className="absolute inset-0 bg-white/10 mix-blend-overlay" />}
    </button>
  );

  return (
    <div className="min-h-screen bg-black text-zinc-100 flex font-sans selection:bg-indigo-500/30">
      
      <aside className="hidden lg:flex fixed inset-y-0 left-0 z-40 w-72 bg-[#0a0a0a] border-r border-white/5 flex-col">
        <div className="p-8 flex items-center gap-3 font-bold text-2xl text-white tracking-tight">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center">
            <img src='/logo.png' alt="Logo"></img>
          </div>
          ObjectDet
        </div>
        
        <nav className="flex-1 px-4 space-y-1 mt-6">
          <div className="text-xs font-bold text-zinc-600 uppercase tracking-widest px-4 mb-4 mt-2">Main Menu</div>
          <NavItem id="dashboard" icon={LayoutDashboard} label="Dashboard" />
          <NavItem id="history" icon={History} label="History" />
          <NavItem id="analytics" icon={BarChart3} label="Analytics" />
          
          <div className="text-xs font-bold text-zinc-600 uppercase tracking-widest px-4 mb-4 mt-8">System</div>
          <NavItem id="specs" icon={Cpu} label="Model Specs" />
          <NavItem id="teams" icon={Users} label="Research Team" />
          <NavItem id="profile" icon={User} label="Settings" />
        </nav>
        
        <div className="p-6 border-t border-white/5">
          <button onClick={()=>{setToken(null);localStorage.removeItem('token');localStorage.removeItem('userEmail')}} className="w-full flex items-center justify-center gap-2 px-4 py-3 text-red-400 hover:bg-red-500/10 rounded-xl transition-colors text-sm font-medium">
            <LogOut size={16}/> Sign Out
          </button>
        </div>
      </aside>
      
      <div className={cn(
        "fixed inset-0 z-50 lg:hidden pointer-events-none transition-all duration-300", 
        isMobileMenuOpen ? "bg-black/80 backdrop-blur-sm pointer-events-auto" : "bg-transparent"
      )} onClick={()=>setIsMobileMenuOpen(false)}>
        <aside 
            onClick={(e)=>e.stopPropagation()}
            className={cn(
                "absolute inset-y-0 left-0 w-[280px] bg-[#0f0f0f] border-r border-white/10 flex flex-col shadow-2xl transition-transform duration-300",
                isMobileMenuOpen ? "translate-x-0" : "-translate-x-full"
            )}
        >
            <div className="p-6 flex items-center justify-between border-b border-white/5">
                <div className="font-bold text-xl text-white flex items-center gap-2">
                    <div className="w-8 h-8 rounded-lg flex items-center justify-center">
                        <img src='/logo.png' alt="Logo"></img>
                    </div>
                    ObjectDet
                </div>
                <button onClick={()=>setIsMobileMenuOpen(false)} className="p-2 hover:bg-white/10 rounded-full text-zinc-400">
                    <X size={20}/>
                </button>
            </div>
            <nav className="flex-1 px-4 py-6 space-y-1 overflow-y-auto">
                <NavItem id="dashboard" icon={LayoutDashboard} label="Dashboard" />
                <NavItem id="history" icon={History} label="History" />
                <NavItem id="analytics" icon={BarChart3} label="Analytics" />
                <div className="my-4 h-px bg-white/5 mx-4"/>
                <NavItem id="specs" icon={Cpu} label="Model Specs" />
                <NavItem id="teams" icon={Users} label="Research Team" />
                <NavItem id="profile" icon={User} label="Settings" />
            </nav>
            <div className="p-4 border-t border-white/5 bg-black/20">
                <button onClick={()=>{setToken(null);localStorage.removeItem('token');localStorage.removeItem('userEmail')}} className="w-full flex items-center gap-3 px-4 py-3 text-red-400 bg-red-500/5 rounded-xl text-sm font-medium">
                    <LogOut size={16}/> Sign Out
                </button>
            </div>
        </aside>
      </div>

      <main className="flex-1 lg:ml-72 min-h-screen flex flex-col relative overflow-hidden bg-black">
        <div className="absolute top-0 left-0 w-full h-[500px] bg-gradient-to-b from-indigo-900/10 to-transparent pointer-events-none" />
        
        <header className="sticky top-0 z-30 lg:hidden bg-black/80 backdrop-blur-xl border-b border-white/5 px-4 h-16 flex items-center justify-between">
            <div className="flex items-center gap-3">
                <button onClick={()=>setIsMobileMenuOpen(true)} className="p-2 -ml-2 hover:bg-white/10 rounded-lg text-white">
                    <Menu size={24}/>
                </button>
                <span className="font-semibold text-white capitalize">{activeTab}</span>
            </div>
            <div className="w-8 h-8 rounded-full bg-zinc-800 overflow-hidden border border-white/10">
                <img src={userProfile.avatar} className="w-full h-full object-cover" alt="User"/>
            </div>
        </header>

        <header className="hidden lg:flex px-8 py-6 justify-between items-center relative z-20">
            <div>
                <h1 className="text-3xl font-bold text-white tracking-tight capitalize">{activeTab === 'specs' ? 'System Specifications' : activeTab}</h1>
                <p className="text-zinc-500 text-sm mt-1">AI-Powered Object Detection</p>
            </div>
            <div className="flex items-center gap-4">
                <div className="text-right hidden xl:block">
                    <div className="text-sm font-bold text-white">{userProfile.name}</div>
                    <div className="text-xs text-zinc-500">{userProfile.email}</div>
                </div>
                <div className="w-10 h-10 rounded-full bg-zinc-800 overflow-hidden border-2 border-zinc-700 shadow-xl">
                    <img src={userProfile.avatar} className="w-full h-full object-cover" alt="User"/>
                </div>
            </div>
        </header>

        <div className="flex-1 p-4 md:p-8 overflow-y-auto relative z-10">
            <AnimatePresence mode="wait">
                <motion.div 
                    key={activeTab} 
                    initial={{opacity:0, y:10}} 
                    animate={{opacity:1, y:0}} 
                    exit={{opacity:0, y:-10}} 
                    transition={{duration:0.3}} 
                    className="max-w-7xl mx-auto pb-10"
                >
                    {activeTab === 'dashboard' && (
                    <DashboardView 
                        selectedFile={selectedFile} previewUrl={previewUrl} predictionResult={predictionResult}
                        isLoading={isLoading} handleFileSelect={handleFileSelect} handlePredict={handlePredict}
                        handleReset={handleReset}
                    />
                    )}
                    {activeTab === 'specs' && <SpecsView />}
                    {activeTab === 'teams' && <TeamsView />}
                    {activeTab === 'analytics' && <AnalyticsView />}
                    {activeTab === 'history' && <HistoryView history={history} fetchHistory={fetchHistory} />}
                    {activeTab === 'profile' && (
                    <ProfileView 
                        userProfile={userProfile} 
                        setUserProfile={setUserProfile} 
                        onAvatarUpload={handleAvatarUpload} 
                        onChangePassword={handleChangePassword}
                        isLoading={isLoading}
                    />
                    )}
                </motion.div>
            </AnimatePresence>
        </div>
      </main>
    </div>
  );
}