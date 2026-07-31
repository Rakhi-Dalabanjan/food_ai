import { Link, useLocation } from 'react-router-dom';
import { Home, Edit3, FileSpreadsheet, Apple } from 'lucide-react';

const Layout = ({ children }) => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Home', icon: <Home size={20} /> },
    { path: '/manual', label: 'Manual Prediction', icon: <Edit3 size={20} /> },
    { path: '/csv', label: 'CSV Prediction', icon: <FileSpreadsheet size={20} /> },
  ];

  return (
    <div className="flex h-screen bg-background overflow-hidden dark">
      {/* Sidebar */}
      <aside className="w-64 glass-panel m-4 flex flex-col hidden md:flex border border-white/10">
        <div className="p-6 flex items-center gap-3 border-b border-white/10">
          <div className="bg-primary/20 p-2 rounded-lg text-primary">
            <Apple size={24} />
          </div>
          <h1 className="font-bold text-xl tracking-tight text-white">ShelfLife AI</h1>
        </div>
        
        <nav className="flex-1 p-4 space-y-2">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                location.pathname === item.path
                  ? 'bg-primary/20 text-primary font-medium'
                  : 'text-slate-400 hover:bg-white/5 hover:text-white'
              }`}
            >
              {item.icon}
              {item.label}
            </Link>
          ))}
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-full overflow-hidden p-4 md:pl-0">
        <div className="flex-1 overflow-y-auto glass-panel p-6 border border-white/10">
          {children}
        </div>
      </main>
    </div>
  );
};

export default Layout;
