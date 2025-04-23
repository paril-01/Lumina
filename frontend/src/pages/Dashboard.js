import React, { useState, useEffect, useContext } from 'react';
import { 
  Box, 
  Grid, 
  Paper, 
  Typography, 
  Button, 
  Card, 
  CardContent, 
  CircularProgress,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider
} from '@mui/material';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import DonutLargeIcon from '@mui/icons-material/DonutLarge';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import { useNavigate } from 'react-router-dom';
import AuthContext from '../context/AuthContext';
import AlertContext from '../context/AlertContext';

function Dashboard() {
  const { user } = useContext(AuthContext);
  const { showAlert } = useContext(AlertContext);
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    learningProgress: 28,
    activeTasks: 3,
    monitoredApps: 12,
    detectedPatterns: 7
  });
  const [recentActivities, setRecentActivities] = useState([]);
  const [systemStatus, setSystemStatus] = useState({
    monitoringActive: true,
    learningActive: true,
    automationActive: false,
    errors: []
  });

  useEffect(() => {
    // Simulate data loading
    const timer = setTimeout(() => {
      // In a real app, we would fetch this data from our backend
      setRecentActivities([
        { 
          id: 1, 
          title: 'New pattern detected', 
          description: 'AI detected you frequently open email after calendar notifications',
          timestamp: '10 minutes ago',
          type: 'pattern'
        },
        { 
          id: 2, 
          title: 'Task automation suggestion', 
          description: 'Would you like to automate formatting in spreadsheets?',
          timestamp: '1 hour ago',
          type: 'suggestion'
        },
        { 
          id: 3, 
          title: 'Learning milestone reached', 
          description: 'Communication style model now has 75% confidence',
          timestamp: '3 hours ago',
          type: 'milestone'
        },
        { 
          id: 4, 
          title: 'Privacy alert', 
          description: 'Sensitive data detected in monitoring. Data was filtered.',
          timestamp: '1 day ago',
          type: 'alert'
        },
      ]);
      
      setLoading(false);
    }, 1500);
    
    return () => clearTimeout(timer);
  }, []);

  const handleStartMonitoring = () => {
    if (!systemStatus.monitoringActive) {
      setSystemStatus({...systemStatus, monitoringActive: true});
      showAlert('Activity monitoring started', 'success');
    }
  };

  const handlePauseMonitoring = () => {
    if (systemStatus.monitoringActive) {
      setSystemStatus({...systemStatus, monitoringActive: false});
      showAlert('Activity monitoring paused', 'info');
    }
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Welcome back, {user?.full_name || user?.username || 'User'}
      </Typography>
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <>
          {/* Status Cards */}
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Paper 
                elevation={3} 
                sx={{ 
                  p: 2, 
                  display: 'flex', 
                  flexDirection: 'column', 
                  alignItems: 'center',
                  height: '100%' 
                }}
              >
                <DonutLargeIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
                <Typography variant="h5" component="div" align="center">
                  {stats.learningProgress}%
                </Typography>
                <Typography variant="body2" color="textSecondary" align="center">
                  Learning Progress
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={stats.learningProgress} 
                  sx={{ width: '100%', mt: 1 }}
                />
                <Button 
                  variant="text" 
                  color="primary" 
                  sx={{ mt: 1 }}
                  onClick={() => navigate('/learning')}
                >
                  View Details
                </Button>
              </Paper>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Paper 
                elevation={3} 
                sx={{ 
                  p: 2, 
                  display: 'flex', 
                  flexDirection: 'column', 
                  alignItems: 'center',
                  height: '100%'
                }}
              >
                <AutoAwesomeIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
                <Typography variant="h5" component="div" align="center">
                  {stats.activeTasks}
                </Typography>
                <Typography variant="body2" color="textSecondary" align="center">
                  Active Automations
                </Typography>
                <Button 
                  variant="text" 
                  color="primary" 
                  sx={{ mt: 1 }}
                  onClick={() => navigate('/tasks')}
                >
                  Manage Tasks
                </Button>
              </Paper>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Paper 
                elevation={3} 
                sx={{ 
                  p: 2, 
                  display: 'flex', 
                  flexDirection: 'column', 
                  alignItems: 'center',
                  height: '100%'
                }}
              >
                <AccessTimeIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
                <Typography variant="h5" component="div" align="center">
                  {stats.monitoredApps}
                </Typography>
                <Typography variant="body2" color="textSecondary" align="center">
                  Monitored Applications
                </Typography>
                <Button 
                  variant="text" 
                  color="primary" 
                  sx={{ mt: 1 }}
                  onClick={() => navigate('/activity')}
                >
                  View Activities
                </Button>
              </Paper>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Paper 
                elevation={3} 
                sx={{ 
                  p: 2, 
                  display: 'flex', 
                  flexDirection: 'column', 
                  alignItems: 'center',
                  height: '100%'
                }}
              >
                <TrendingUpIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
                <Typography variant="h5" component="div" align="center">
                  {stats.detectedPatterns}
                </Typography>
                <Typography variant="body2" color="textSecondary" align="center">
                  Detected Patterns
                </Typography>
                <Button 
                  variant="text" 
                  color="primary" 
                  sx={{ mt: 1 }}
                  onClick={() => navigate('/learning')}
                >
                  View Patterns
                </Button>
              </Paper>
            </Grid>
          </Grid>
          
          {/* Main Content */}
          <Grid container spacing={3}>
            {/* System Status */}
            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    System Status
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemIcon>
                        {systemStatus.monitoringActive ? 
                          <CheckCircleIcon color="success" /> : 
                          <ErrorIcon color="error" />
                        }
                      </ListItemIcon>
                      <ListItemText 
                        primary="Activity Monitoring" 
                        secondary={systemStatus.monitoringActive ? "Active" : "Paused"} 
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        {systemStatus.learningActive ? 
                          <CheckCircleIcon color="success" /> : 
                          <ErrorIcon color="error" />
                        }
                      </ListItemIcon>
                      <ListItemText 
                        primary="Learning Engine" 
                        secondary={systemStatus.learningActive ? "Learning" : "Paused"} 
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        {systemStatus.automationActive ? 
                          <CheckCircleIcon color="success" /> : 
                          <ErrorIcon color="warning" />
                        }
                      </ListItemIcon>
                      <ListItemText 
                        primary="Task Automation" 
                        secondary={systemStatus.automationActive ? "Active" : "Disabled"} 
                      />
                    </ListItem>
                  </List>
                  <Divider sx={{ my: 2 }} />
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Button 
                      variant="contained" 
                      color="primary" 
                      onClick={handleStartMonitoring}
                      disabled={systemStatus.monitoringActive}
                    >
                      Start
                    </Button>
                    <Button 
                      variant="outlined" 
                      color="secondary" 
                      onClick={handlePauseMonitoring}
                      disabled={!systemStatus.monitoringActive}
                    >
                      Pause
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            
            {/* Recent Activities */}
            <Grid item xs={12} md={8}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Recent Activities & Insights
                  </Typography>
                  <List>
                    {recentActivities.map((activity) => (
                      <React.Fragment key={activity.id}>
                        <ListItem alignItems="flex-start">
                          <ListItemIcon>
                            {activity.type === 'pattern' && <TrendingUpIcon color="primary" />}
                            {activity.type === 'suggestion' && <AutoAwesomeIcon color="secondary" />}
                            {activity.type === 'milestone' && <DonutLargeIcon color="success" />}
                            {activity.type === 'alert' && <ErrorIcon color="error" />}
                          </ListItemIcon>
                          <ListItemText
                            primary={activity.title}
                            secondary={
                              <>
                                <Typography
                                  component="span"
                                  variant="body2"
                                  color="textPrimary"
                                >
                                  {activity.description}
                                </Typography>
                                {" — "}{activity.timestamp}
                              </>
                            }
                          />
                        </ListItem>
                        <Divider variant="inset" component="li" />
                      </React.Fragment>
                    ))}
                  </List>
                  <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
                    <Button 
                      variant="text" 
                      color="primary"
                      onClick={() => navigate('/activity')}
                    >
                      View All Activities
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </>
      )}
    </Box>
  );
}

export default Dashboard;
