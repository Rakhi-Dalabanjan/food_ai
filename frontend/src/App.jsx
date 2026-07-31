import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import ManualPrediction from './pages/ManualPrediction';
import CsvPrediction from './pages/CsvPrediction';
import ModelMetrics from './pages/ModelMetrics';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/manual" element={<ManualPrediction />} />
          <Route path="/csv" element={<CsvPrediction />} />
          <Route path="/metrics" element={<ModelMetrics />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
