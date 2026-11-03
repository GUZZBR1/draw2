import { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import { Navigate, Route, Routes, useLocation } from 'react-router-dom';
import ScrollbarControl from '../components/ScrollbarControl';
import SiteHeader from '../components/SiteHeader';
import HomePage from '../pages/HomePage';
import ProductPage from '../pages/ProductPage';
import VisionPage from '../pages/VisionPage';

export default function App() {
  const location = useLocation();
  const [productRefresh, setProductRefresh] = useState(0);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'auto' });
    document.body.classList.toggle('home-scroll', location.pathname === '/');
  }, [location.pathname]);

  return (
    <>
      <SiteHeader
        onRepeatProduct={() => {
          setProductRefresh((revision) => revision + 1);
          window.scrollTo({ top: 0, behavior: 'auto' });
        }}
      />
      <motion.div
        className="route-transition"
        key={location.pathname}
        initial={false}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.28, ease: 'easeOut' }}
      >
        <Routes location={location}>
          <Route path="/" element={<HomePage />} />
          <Route path="/vision" element={<VisionPage />} />
          <Route path="/product" element={<ProductPage key={productRefresh} />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </motion.div>
      <ScrollbarControl />
    </>
  );
}
