import React from 'react';
import { Box, Typography, Button, Paper } from '@mui/material';
import ErrorIcon from '@mui/icons-material/Error';
import RefreshIcon from '@mui/icons-material/Refresh';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { 
      hasError: false,
      error: null,
      errorInfo: null
    };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // Log the error to an error reporting service
    console.error("Error caught by ErrorBoundary:", error, errorInfo);
    this.setState({ errorInfo });
    
    // You could also send the error to your analytics service here
    if (process.env.NODE_ENV === 'production') {
      // Example: Send to error reporting service
      // errorReportingService.captureException(error, { extra: errorInfo });
    }
  }

  handleReload = () => {
    window.location.reload();
  }

  handleGoHome = () => {
    window.location.href = '/';
  }

  render() {
    if (this.state.hasError) {
      const { error, errorInfo } = this.state;
      const errorMessage = error?.message || 'An unexpected error occurred';
      
      // You can render any custom fallback UI
      return (
        <Box 
          sx={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center',
            minHeight: '100vh',
            p: 3,
            bgcolor: 'background.default'
          }}
        >
          <Paper 
            elevation={3} 
            sx={{ 
              p: 4, 
              maxWidth: 600, 
              width: '100%',
              borderRadius: 2,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              textAlign: 'center'
            }}
          >
            <ErrorIcon color="error" sx={{ fontSize: 70, mb: 2 }} />
            
            <Typography variant="h5" component="h1" gutterBottom color="error" fontWeight="bold">
              Something went wrong
            </Typography>
            
            <Typography variant="body1" color="textSecondary" sx={{ mb: 3 }}>
              {errorMessage}
            </Typography>
            
            <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
              <Button 
                variant="contained" 
                color="primary" 
                onClick={this.handleReload}
                startIcon={<RefreshIcon />}
              >
                Refresh Page
              </Button>
              
              <Button 
                variant="outlined" 
                color="primary" 
                onClick={this.handleGoHome}
              >
                Go to Home
              </Button>
            </Box>
            
            {process.env.NODE_ENV !== 'production' && errorInfo && (
              <Box sx={{ mt: 4, textAlign: 'left', width: '100%', overflow: 'auto' }}>
                <Typography variant="subtitle2" gutterBottom>
                  Error Details (Development Mode Only):
                </Typography>
                <Box 
                  component="pre" 
                  sx={{ 
                    p: 2, 
                    bgcolor: 'grey.900', 
                    color: 'grey.300', 
                    borderRadius: 1,
                    overflow: 'auto',
                    fontSize: '0.75rem',
                    maxHeight: '200px'
                  }}
                >
                  <Box component="code">
                    {error && error.toString()}
                    {errorInfo && `\n\nComponent Stack:\n${errorInfo.componentStack}`}
                  </Box>
                </Box>
              </Box>
            )}
          </Paper>
        </Box>
      );
    }

    // If there's no error, render children normally
    return this.props.children;
  }
}

export default ErrorBoundary;
