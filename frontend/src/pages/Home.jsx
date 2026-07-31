import { Link } from 'react-router-dom';
import { Edit3, FileSpreadsheet, ArrowRight, BarChart2 } from 'lucide-react';

const Home = () => {
  return (
    <div className="space-y-12 animate-in fade-in duration-500 max-w-5xl mx-auto py-8">
      <div className="text-center space-y-4">
        <h1 className="text-5xl font-extrabold bg-gradient-to-r from-primary to-emerald-400 bg-clip-text text-transparent">
          ShelfLife AI
        </h1>
        <p className="text-slate-400 text-xl font-medium">
          Predict Food Shelf Life Using Machine Learning
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <Link 
          to="/manual" 
          className="group bg-surface/50 p-8 rounded-2xl border border-white/10 hover:border-primary/50 transition-all hover:shadow-lg hover:shadow-primary/10 flex flex-col items-center text-center space-y-4"
        >
          <div className="bg-primary/20 p-5 rounded-full text-primary group-hover:scale-110 transition-transform">
            <Edit3 size={40} />
          </div>
          <h2 className="text-2xl font-bold">Manual Prediction</h2>
          <p className="text-slate-400">
            Enter specific food parameters manually to get an instant AI prediction on remaining shelf life.
          </p>
          <div className="text-primary font-medium flex items-center gap-2 mt-4 group-hover:gap-3 transition-all">
            Start Manual Prediction <ArrowRight size={18} />
          </div>
        </Link>

        <Link 
          to="/csv" 
          className="group bg-surface/50 p-8 rounded-2xl border border-white/10 hover:border-emerald-500/50 transition-all hover:shadow-lg hover:shadow-emerald-500/10 flex flex-col items-center text-center space-y-4"
        >
          <div className="bg-emerald-500/20 p-5 rounded-full text-emerald-500 group-hover:scale-110 transition-transform">
            <FileSpreadsheet size={40} />
          </div>
          <h2 className="text-2xl font-bold">CSV Upload Prediction</h2>
          <p className="text-slate-400">
            Upload bulk data via CSV for batch predictions and download the results instantly.
          </p>
          <div className="text-emerald-500 font-medium flex items-center gap-2 mt-4 group-hover:gap-3 transition-all">
            Upload CSV File <ArrowRight size={18} />
          </div>
        </Link>
      </div>

      <div className="bg-slate-900/50 rounded-2xl border border-white/5 p-8 mt-12 text-center">
        <h3 className="text-2xl font-bold mb-8">How It Works</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 relative">
          <div className="hidden md:block absolute top-1/2 left-1/6 right-1/6 h-0.5 bg-white/5 -z-10"></div>
          
          <div className="flex flex-col items-center space-y-3">
            <div className="w-12 h-12 rounded-full bg-slate-800 border-4 border-background flex items-center justify-center font-bold text-xl z-10">1</div>
            <h4 className="font-semibold text-lg">Enter Parameters</h4>
            <p className="text-slate-400 text-sm">Provide food quality data manually or via CSV.</p>
          </div>
          
          <div className="flex flex-col items-center space-y-3">
            <div className="w-12 h-12 rounded-full bg-slate-800 border-4 border-background flex items-center justify-center font-bold text-xl z-10">2</div>
            <h4 className="font-semibold text-lg">AI Analyzes Data</h4>
            <p className="text-slate-400 text-sm">Our Random Forest model processes the input instantly.</p>
          </div>
          
          <div className="flex flex-col items-center space-y-3">
            <div className="w-12 h-12 rounded-full bg-primary/20 text-primary border-4 border-background flex items-center justify-center font-bold text-xl z-10">3</div>
            <h4 className="font-semibold text-lg">Get Prediction</h4>
            <p className="text-slate-400 text-sm">View shelf life and calculated spoilage risk level.</p>
          </div>
        </div>
      </div>
      
      <div className="flex justify-center pt-4">
        <Link to="/metrics" className="flex items-center gap-2 text-slate-500 hover:text-slate-300 transition-colors text-sm">
          <BarChart2 size={16} /> View Model Metrics
        </Link>
      </div>
    </div>
  );
};

export default Home;
