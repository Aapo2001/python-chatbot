import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Home from './pages/Home'
import QuickStart from './pages/QuickStart'
import Architecture from './pages/Architecture'
import Configuration from './pages/Configuration'
import Ros2 from './pages/Ros2'
import ApiReference from './pages/ApiReference'

function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="quick-start" element={<QuickStart />} />
        <Route path="architecture" element={<Architecture />} />
        <Route path="configuration" element={<Configuration />} />
        <Route path="ros2" element={<Ros2 />} />
        <Route path="api-reference" element={<ApiReference />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}

export default App
