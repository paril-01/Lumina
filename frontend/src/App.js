import React, { useState, useEffect, useMemo } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { alpha } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// MUI Components
import Box from '@mui/material/Box';
import Alert from '@mui/material/Alert';
import Snackbar from '@mui/material/Snackbar';
import Paper from '@mui/material/Paper';
import Backdrop from '@mui/material/Backdrop';
import CircularProgress from '@mui/material/CircularProgress';

// Custom Components
import Dashboard from './pages/Dashboard';
import Login from './pages/Login';
import Settings from './pages/Settings';
import TaskManager from './pages/TaskManager';
import ActivityMonitor from './pages/ActivityMonitor';
import LearningProgress from './pages/LearningProgress';
import ChatInterface from './pages/ChatInterface';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import PrivacyPolicy from './pages/PrivacyPolicy';

// Context
import AuthContext from './context/AuthContext';
import AlertContext from './context/AlertContext';
import ThemeContext from './context/ThemeContext';

// Define futuristic theme options for light and dark modes
const getThemeOptions = (mode) => ({
  breakpoints: {
    values: {
      xs: 0,
      sm: 600,
      md: 960,
      lg: 1280,
      xl: 1920,
    },
  },
  palette: {
    mode,
    primary: {
      main: mode === 'dark' ? '#5e35b1' : '#3f51b5',
      light: mode === 'dark' ? '#9162e4' : '#7986cb',
      dark: mode === 'dark' ? '#4527a0' : '#303f9f',
      contrastText: '#ffffff',
    },
    secondary: {
      main: mode === 'dark' ? '#00b0ff' : '#f50057',
      light: mode === 'dark' ? '#69e2ff' : '#ff4081',
      dark: mode === 'dark' ? '#0081cb' : '#c51162',
      contrastText: '#ffffff',
    },
    background: {
      default: mode === 'dark' ? '#121212' : '#f5f5f5',
      paper: mode === 'dark' ? '#1e1e1e' : '#ffffff',
      sidebar: mode === 'dark' ? '#0a0a0a' : '#f0f0f0',
      card: mode === 'dark' ? '#252525' : '#ffffff',
      dialog: mode === 'dark' ? '#2d2d2d' : '#ffffff',
    },
    text: {
      primary: mode === 'dark' ? '#e0e0e0' : '#212121',
      secondary: mode === 'dark' ? '#aaaaaa' : '#757575',
      disabled: mode === 'dark' ? '#666666' : '#9e9e9e',
      hint: mode === 'dark' ? '#666666' : '#9e9e9e',
    },
    error: {
      main: '#f44336',
    },
    warning: {
      main: '#ff9800',
    },
    info: {
      main: mode === 'dark' ? '#29b6f6' : '#2196f3',
    },
    success: {
      main: '#4caf50',
    },
    divider: mode === 'dark' ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)',
  },
  shape: {
    borderRadius: 12,
  },
  shadows: [
    'none',
    '0 2px 4px rgba(0,0,0,0.1)',
    '0 4px 8px rgba(0,0,0,0.12)',
    '0 8px 16px rgba(0,0,0,0.12)',
    '0 12px 24px rgba(0,0,0,0.14)',
    '0 16px 32px rgba(0,0,0,0.14)',
    '0 20px 40px rgba(0,0,0,0.16)',
    ...Array(18).fill('none'),
  ],
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
      letterSpacing: '-0.01562em',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      letterSpacing: '-0.00833em',
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
      letterSpacing: '0',
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
      letterSpacing: '0.00735em',
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
      letterSpacing: '0',
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
      letterSpacing: '0.0075em',
    },
    button: {
      textTransform: 'none',
      fontWeight: 500,
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        '*': {
          boxSizing: 'border-box',
        },
        html: {
          MozOsxFontSmoothing: 'grayscale',
          WebkitFontSmoothing: 'antialiased',
          height: '100%',
          width: '100%',
        },
        body: {
          height: '100%',
          width: '100%',
          transition: 'background-color 0.3s ease, color 0.3s ease',
          scrollbarWidth: 'thin',
          '&::-webkit-scrollbar': {
            width: '8px',
            height: '8px',
          },
          '&::-webkit-scrollbar-thumb': {
            backgroundColor: mode === 'dark' ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)',
            borderRadius: '4px',
          },
          '&::-webkit-scrollbar-track': {
            backgroundColor: mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)',
          },
        },
        '#root': {
          height: '100%',
          width: '100%',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          backgroundImage: mode === 'dark' ? 
            'linear-gradient(145deg, rgba(40,40,40,0.95), rgba(30,30,30,0.95))' : 
            'linear-gradient(145deg, rgba(255,255,255,0.95), rgba(240,240,240,0.95))',
          backdropFilter: 'blur(10px)',
          boxShadow: mode === 'dark' ? 
            '5px 5px 10px rgba(0,0,0,0.2), -5px -5px 10px rgba(50,50,50,0.1)' : 
            '5px 5px 10px rgba(0,0,0,0.05), -5px -5px 10px rgba(255,255,255,0.6)',
          transition: 'all 0.3s ease',
          border: mode === 'dark' ? '1px solid rgba(255,255,255,0.1)' : '1px solid rgba(0,0,0,0.05)',
          '&:hover': {
            boxShadow: mode === 'dark' ? 
              '8px 8px 16px rgba(0,0,0,0.3), -8px -8px 16px rgba(50,50,50,0.1)' : 
              '8px 8px 16px rgba(0,0,0,0.08), -8px -8px 16px rgba(255,255,255,0.8)',
            transform: 'translateY(-2px)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: mode === 'dark' ? 
            'linear-gradient(145deg, rgba(40,40,40,0.95), rgba(30,30,30,0.95))' : 
            'linear-gradient(145deg, rgba(255,255,255,0.95), rgba(240,240,240,0.95))',
          backdropFilter: 'blur(10px)',
          borderRadius: 12,
          border: mode === 'dark' ? '1px solid rgba(255,255,255,0.1)' : '1px solid rgba(0,0,0,0.05)',
        },
      },
    },
    MuiButton: {
      defaultProps: {
        disableElevation: true,
      },
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
          borderRadius: 8,
        },
        contained: {
          position: 'relative',
          overflow: 'hidden',
          padding: '10px 20px',
          backgroundImage: mode === 'dark' ?
            `linear-gradient(135deg, ${alpha('#5e35b1', 0.9)}, ${alpha('#4527a0', 0.9)})` :
            `linear-gradient(135deg, ${alpha('#3f51b5', 0.9)}, ${alpha('#303f9f', 0.9)})`,
          boxShadow: mode === 'dark' ?
            `0 4px 12px ${alpha('#5e35b1', 0.4)}` :
            `0 4px 12px ${alpha('#3f51b5', 0.3)}`,
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: '-100%',
            width: '100%',
            height: '100%',
            background: `linear-gradient(90deg, transparent, ${alpha('#ffffff', 0.2)}, transparent)`,
            transition: 'left 0.5s ease',
          },
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: mode === 'dark' ?
              `0 8px 16px ${alpha('#5e35b1', 0.5)}` :
              `0 8px 16px ${alpha('#3f51b5', 0.4)}`,
            '&::before': {
              left: '100%',
            },
          },
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: mode === 'dark' ? '#0a0a0a' : '#f0f0f0',
          borderRight: `1px solid ${mode === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
        },
      },
    },
  },
});

function App() {
  const [user, setUser] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [alert, setAlert] = useState({ open: false, message: '', severity: 'info' });
  const [loading, setLoading] = useState(false);
  
  // Use dark mode by default
  const [themeMode, setThemeMode] = useState(() => {
    // Check local storage for theme preference
    const savedMode = localStorage.getItem('lumina_theme_mode');
    return savedMode || 'dark';
  });
  
  // Create a futuristic theme based on mode
  const theme = useMemo(() => createTheme(getThemeOptions(themeMode)), [themeMode]);
  
  // Toggle theme function
  const toggleTheme = () => {
    const newMode = themeMode === 'light' ? 'dark' : 'light';
    setThemeMode(newMode);
    localStorage.setItem('lumina_theme_mode', newMode);
  };

  // Check for saved user session on load
  useEffect(() => {
    const savedUser = localStorage.getItem('lumina_user');
    if (savedUser) {
      try {
        setUser(JSON.parse(savedUser));
      } catch (error) {
        console.error('Error parsing saved user:', error);
        localStorage.removeItem('lumina_user');
      }
    }
  }, []);

  // Show alert message
  const showAlert = (message, severity = 'info') => {
    setAlert({
      open: true,
      message,
      severity,
    });
  };

  // Handle alert close
  const handleAlertClose = () => {
    setAlert({
      ...alert,
      open: false,
    });
  };

  // Toggle sidebar
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ThemeContext.Provider value={{ mode: themeMode, toggleTheme }}>
        <AlertContext.Provider value={{ showAlert }}>
          <AuthContext.Provider value={{ user, setUser }}>
            <Router>
              <Box sx={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
                {user && (
                  <>
                    <Header 
                      sidebarOpen={sidebarOpen} 
                      toggleSidebar={toggleSidebar} 
                    />
                    <Sidebar 
                      open={sidebarOpen} 
                      toggleDrawer={toggleSidebar} 
                    />
                  </>
                )}
                
                <Box
                  component="main"
                  sx={{
                    flexGrow: 1,
                    p: 3,
                    width: '100%',
                    height: '100%',
                    overflow: 'auto',
                    pt: user ? 8 : 3,
                    pl: user ? (sidebarOpen ? { sm: 32, xs: 3 } : 3) : 3,
                    transition: theme => theme.transitions.create(['margin', 'width'], {
                      easing: theme.transitions.easing.sharp,
                      duration: theme.transitions.duration.leavingScreen,
                    }),
                  }}
                >
                  <Routes>
                    {/* Make ChatInterface the default landing page */}
                    <Route path="/" element={user ? <ChatInterface /> : <Navigate to="/login" />} />
                    <Route path="/dashboard" element={user ? <Dashboard /> : <Navigate to="/login" />} />
                    <Route path="/tasks" element={user ? <TaskManager /> : <Navigate to="/login" />} />
                    <Route path="/activities" element={user ? <ActivityMonitor /> : <Navigate to="/login" />} />
                    <Route path="/learning" element={user ? <LearningProgress /> : <Navigate to="/login" />} />
                    <Route path="/settings" element={user ? <Settings /> : <Navigate to="/login" />} />
                    <Route path="/privacy" element={user ? <PrivacyPolicy /> : <Navigate to="/login" />} />
                    <Route path="/chat" element={user ? <ChatInterface /> : <Navigate to="/login" />} />
                    <Route path="/login" element={!user ? <Login /> : <Navigate to="/" />} />
                  </Routes>
                </Box>
              </Box>
            </Router>
          </AuthContext.Provider>
        </AlertContext.Provider>
      </ThemeContext.Provider>
      
      <Snackbar 
        open={alert.open} 
        autoHideDuration={6000} 
        onClose={handleAlertClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleAlertClose} severity={alert.severity} sx={{ width: '100%' }}>
          {alert.message}
        </Alert>
      </Snackbar>
      
      <Backdrop
        sx={{
          zIndex: theme => theme.zIndex.drawer + 1,
          color: '#fff',
          backdropFilter: 'blur(4px)',
        }}
        open={loading}
      >
        <CircularProgress color="primary" />
      </Backdrop>
    </ThemeProvider>
  );
}

export default App;
