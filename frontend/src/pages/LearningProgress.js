import React, { useState, useEffect, useContext } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  CircularProgress,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Divider,
  Chip,
  Tabs,
  Tab,
  LinearProgress,
  Button,
  Alert
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import PsychologyIcon from '@mui/icons-material/Psychology';
import PersonIcon from '@mui/icons-material/Person';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import SchoolIcon from '@mui/icons-material/School';
import LightbulbIcon from '@mui/icons-material/Lightbulb';
import AppsIcon from '@mui/icons-material/Apps';
import AlertContext from '../context/AlertContext';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, ArcElement, BarElement } from 'chart.js';
import { Line, Pie, Bar } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, ArcElement, BarElement);

function LearningProgress() {
  const { showAlert } = useContext(AlertContext);
  const [loading, setLoading] = useState(true);
  const [tabValue, setTabValue] = useState(0);
  const [selectedPattern, setSelectedPattern] = useState(null);
  const [learningData, setLearningData] = useState({
    stage: 'active',
    progress: 42,
    patterns: [],
    activities: 0,
    applicationUsage: {},
    confidenceHistory: []
  });

  useEffect(() => {
    // Simulate loading learning data from API
    const timer = setTimeout(() => {
      const mockData = generateMockLearningData();
      setLearningData(mockData);
      setLoading(false);
    }, 1500);
    
    return () => clearTimeout(timer);
  }, []);

  const generateMockLearningData = () => {
    // Generate mock data for the learning progress
    const applications = ['Chrome', 'Word', 'Excel', 'Outlook', 'Teams', 'Visual Studio', 'File Explorer', 'Slack'];
    const appUsage = {};
    applications.forEach(app => {
      appUsage[app] = Math.floor(Math.random() * 100);
    });

    // Generate confidence history (last 10 days)
    const confidenceHistory = [];
    const now = new Date();
    for (let i = 9; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      // Confidence generally increases over time with some variation
      const confidence = Math.min(80, 30 + (9-i) * 5 + Math.floor(Math.random() * 10));
      confidenceHistory.push({
        date: date.toISOString().split('T')[0],
        confidence
      });
    }

    // Generate detected patterns
    const patterns = [
      {
        id: 1,
        type: 'application_sequence',
        description: 'Opens email client after checking calendar',
        confidence: 87,
        occurrences: 42,
        first_detected: '2025-04-10T09:23:15',
        last_seen: '2025-04-22T08:45:22',
        details: {
          trigger: 'Calendar app closed',
          action: 'Open Outlook',
          time_pattern: 'Weekdays between 8:00-9:30 AM',
          success_rate: 92
        }
      },
      {
        id: 2,
        type: 'keyboard_pattern',
        description: 'Uses Ctrl+C, Ctrl+V sequence frequently in documents',
        confidence: 94,
        occurrences: 156,
        first_detected: '2025-04-08T14:12:03',
        last_seen: '2025-04-22T10:15:42',
        details: {
          applications: ['Word', 'Excel', 'Notepad'],
          frequency: 'High',
          context: 'Document editing',
          success_rate: 98
        }
      },
      {
        id: 3,
        type: 'temporal_pattern',
        description: 'Starts chat applications after lunch break',
        confidence: 76,
        occurrences: 18,
        first_detected: '2025-04-15T13:05:11',
        last_seen: '2025-04-21T13:02:37',
        details: {
          time_range: '13:00-13:15',
          applications: ['Teams', 'Slack'],
          days: 'Weekdays',
          success_rate: 82
        }
      },
      {
        id: 4,
        type: 'application_usage',
        description: 'Uses Excel for 30+ minutes in the late afternoon',
        confidence: 68,
        occurrences: 12,
        first_detected: '2025-04-12T16:22:18',
        last_seen: '2025-04-22T16:05:42',
        details: {
          time_range: '16:00-17:30',
          duration: '30-45 minutes',
          frequency: 'Weekdays',
          success_rate: 75
        }
      },
      {
        id: 5,
        type: 'communication_style',
        description: 'Uses brief, direct communication in chat applications',
        confidence: 81,
        occurrences: 64,
        first_detected: '2025-04-09T10:45:32',
        last_seen: '2025-04-22T11:12:08',
        details: {
          applications: ['Teams', 'Slack'],
          characteristics: ['Brief messages', 'Few emojis', 'Quick responses'],
          success_rate: 88
        }
      }
    ];

    return {
      stage: 'active',
      progress: 42,
      startDate: '2025-04-08',
      patterns,
      activities: 2457,
      applicationUsage: appUsage,
      confidenceHistory
    };
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handlePatternSelect = (pattern) => {
    setSelectedPattern(pattern);
  };

  const handleCreateAutomation = () => {
    if (selectedPattern) {
      showAlert(`Creating automation for: ${selectedPattern.description}`, 'success');
      // In a real app, this would navigate to the task creation page or open a dialog
    }
  };

  // Chart data for application usage
  const appUsageData = {
    labels: Object.keys(learningData.applicationUsage),
    datasets: [
      {
        label: 'Usage Frequency',
        data: Object.values(learningData.applicationUsage),
        backgroundColor: [
          'rgba(255, 99, 132, 0.5)',
          'rgba(54, 162, 235, 0.5)',
          'rgba(255, 206, 86, 0.5)',
          'rgba(75, 192, 192, 0.5)',
          'rgba(153, 102, 255, 0.5)',
          'rgba(255, 159, 64, 0.5)',
          'rgba(199, 199, 199, 0.5)',
          'rgba(83, 102, 255, 0.5)',
        ],
        borderWidth: 1,
      },
    ],
  };

  // Chart data for confidence history
  const confidenceData = {
    labels: learningData.confidenceHistory.map(item => item.date),
    datasets: [
      {
        label: 'Model Confidence',
        data: learningData.confidenceHistory.map(item => item.confidence),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        tension: 0.2
      }
    ]
  };

  // Chart data for pattern types
  const patternTypesData = {
    labels: ['Application Sequence', 'Keyboard Pattern', 'Temporal Pattern', 'Application Usage', 'Communication Style'],
    datasets: [
      {
        label: 'Pattern Count',
        data: [1, 1, 1, 1, 1], // In a real app, this would count patterns by type
        backgroundColor: [
          'rgba(255, 99, 132, 0.5)',
          'rgba(54, 162, 235, 0.5)',
          'rgba(255, 206, 86, 0.5)',
          'rgba(75, 192, 192, 0.5)',
          'rgba(153, 102, 255, 0.5)',
        ],
        borderColor: [
          'rgb(255, 99, 132)',
          'rgb(54, 162, 235)',
          'rgb(255, 206, 86)',
          'rgb(75, 192, 192)',
          'rgb(153, 102, 255)',
        ],
        borderWidth: 1,
      },
    ],
  };

  // Get pattern icon based on type
  const getPatternIcon = (type) => {
    switch (type) {
      case 'application_sequence':
        return <AppsIcon />;
      case 'keyboard_pattern':
        return <PsychologyIcon />;
      case 'temporal_pattern':
        return <AccessTimeIcon />;
      case 'application_usage':
        return <AppsIcon />;
      case 'communication_style':
        return <PersonIcon />;
      default:
        return <TrendingUpIcon />;
    }
  };

  // Get pattern type label
  const getPatternTypeLabel = (type) => {
    switch (type) {
      case 'application_sequence':
        return 'App Sequence';
      case 'keyboard_pattern':
        return 'Keyboard';
      case 'temporal_pattern':
        return 'Time-based';
      case 'application_usage':
        return 'App Usage';
      case 'communication_style':
        return 'Communication';
      default:
        return type;
    }
  };

  // Format date string
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString();
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Learning Progress
      </Typography>
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <>
          {/* Learning Progress Overview */}
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={12} md={5}>
              <Paper sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <SchoolIcon color="primary" sx={{ fontSize: 40, mr: 2 }} />
                  <Typography variant="h5">Learning Status</Typography>
                </Box>
                
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
                  <Box sx={{ position: 'relative', display: 'inline-flex', mr: 3 }}>
                    <CircularProgress 
                      variant="determinate" 
                      value={learningData.progress} 
                      size={80} 
                      thickness={5} 
                      color="primary"
                    />
                    <Box
                      sx={{
                        top: 0,
                        left: 0,
                        bottom: 0,
                        right: 0,
                        position: 'absolute',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                    >
                      <Typography variant="h6" component="div" color="primary">
                        {`${learningData.progress}%`}
                      </Typography>
                    </Box>
                  </Box>
                  <Box>
                    <Typography variant="h6">
                      {learningData.stage === 'initial' && 'Initial Learning'}
                      {learningData.stage === 'active' && 'Active Learning'}
                      {learningData.stage === 'mature' && 'Mature Model'}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Learning since {formatDate(learningData.startDate)}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {learningData.activities} activities processed
                    </Typography>
                  </Box>
                </Box>
                
                <Divider sx={{ my: 2 }} />
                
                <Typography variant="body1" sx={{ mb: 1 }}>
                  Learning Progress by Category:
                </Typography>
                
                <Box sx={{ mb: 1 }}>
                  <Typography variant="body2">Application Usage</Typography>
                  <LinearProgress variant="determinate" value={85} sx={{ height: 8, borderRadius: 4 }} />
                  <Typography variant="caption" sx={{ display: 'block', textAlign: 'right' }}>85%</Typography>
                </Box>
                
                <Box sx={{ mb: 1 }}>
                  <Typography variant="body2">Keyboard Patterns</Typography>
                  <LinearProgress variant="determinate" value={70} sx={{ height: 8, borderRadius: 4 }} />
                  <Typography variant="caption" sx={{ display: 'block', textAlign: 'right' }}>70%</Typography>
                </Box>
                
                <Box sx={{ mb: 1 }}>
                  <Typography variant="body2">Time Patterns</Typography>
                  <LinearProgress variant="determinate" value={60} sx={{ height: 8, borderRadius: 4 }} />
                  <Typography variant="caption" sx={{ display: 'block', textAlign: 'right' }}>60%</Typography>
                </Box>
                
                <Box sx={{ mb: 1 }}>
                  <Typography variant="body2">Communication Style</Typography>
                  <LinearProgress variant="determinate" value={45} sx={{ height: 8, borderRadius: 4 }} />
                  <Typography variant="caption" sx={{ display: 'block', textAlign: 'right' }}>45%</Typography>
                </Box>
                
                <Box sx={{ flex: 1 }} />
                
                <Alert severity="info" sx={{ mt: 2 }}>
                  AI learning improves with more usage. Continue using your applications normally for better results.
                </Alert>
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={7}>
              <Paper sx={{ p: 2, height: '100%' }}>
                <Tabs 
                  value={tabValue} 
                  onChange={handleTabChange} 
                  sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}
                >
                  <Tab icon={<TrendingUpIcon />} label="Confidence" />
                  <Tab icon={<AppsIcon />} label="App Usage" />
                  <Tab icon={<PsychologyIcon />} label="Pattern Types" />
                </Tabs>
                
                {tabValue === 0 && (
                  <Box sx={{ height: 300, p: 1 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Learning Confidence Over Time
                    </Typography>
                    <Line 
                      data={confidenceData} 
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                          y: {
                            min: 0,
                            max: 100
                          }
                        }
                      }} 
                    />
                  </Box>
                )}
                
                {tabValue === 1 && (
                  <Box sx={{ height: 300, p: 1 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Application Usage Distribution
                    </Typography>
                    <Pie 
                      data={appUsageData} 
                      options={{
                        responsive: true,
                        maintainAspectRatio: false
                      }} 
                    />
                  </Box>
                )}
                
                {tabValue === 2 && (
                  <Box sx={{ height: 300, p: 1 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Detected Pattern Types
                    </Typography>
                    <Bar 
                      data={patternTypesData} 
                      options={{
                        responsive: true,
                        maintainAspectRatio: false
                      }} 
                    />
                  </Box>
                )}
              </Paper>
            </Grid>
          </Grid>
          
          {/* Detected Patterns */}
          <Typography variant="h5" gutterBottom>
            Detected Behavior Patterns
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={5}>
              <Paper sx={{ maxHeight: 400, overflow: 'auto' }}>
                <List>
                  {learningData.patterns.map((pattern) => (
                    <ListItemButton 
                      key={pattern.id} 
                      selected={selectedPattern?.id === pattern.id}
                      onClick={() => handlePatternSelect(pattern)}
                    >
                      <ListItemIcon>
                        {getPatternIcon(pattern.type)}
                      </ListItemIcon>
                      <ListItemText 
                        primary={pattern.description}
                        secondary={
                          <>
                            <Chip 
                              label={getPatternTypeLabel(pattern.type)} 
                              size="small" 
                              sx={{ mr: 1, fontSize: '0.7rem' }} 
                            />
                            <Chip 
                              label={`${pattern.confidence}% confident`} 
                              size="small" 
                              color={pattern.confidence > 80 ? 'success' : 'warning'}
                              sx={{ fontSize: '0.7rem' }} 
                            />
                          </>
                        }
                      />
                    </ListItemButton>
                  ))}
                </List>
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={7}>
              <Paper sx={{ p: 3, height: '100%' }}>
                {selectedPattern ? (
                  <>
                    <Typography variant="h6" gutterBottom>
                      {selectedPattern.description}
                    </Typography>
                    
                    <Box sx={{ display: 'flex', mb: 2 }}>
                      <Chip 
                        icon={getPatternIcon(selectedPattern.type)} 
                        label={getPatternTypeLabel(selectedPattern.type)} 
                        color="primary" 
                        sx={{ mr: 1 }} 
                      />
                      <Chip 
                        icon={<TrendingUpIcon />} 
                        label={`${selectedPattern.confidence}% confidence`} 
                        color={selectedPattern.confidence > 80 ? 'success' : 'warning'} 
                      />
                    </Box>
                    
                    <Grid container spacing={2} sx={{ mb: 2 }}>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          First Detected:
                        </Typography>
                        <Typography variant="body1">
                          {formatDate(selectedPattern.first_detected)}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Last Seen:
                        </Typography>
                        <Typography variant="body1">
                          {formatDate(selectedPattern.last_seen)}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Occurrences:
                        </Typography>
                        <Typography variant="body1">
                          {selectedPattern.occurrences} times
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Success Rate:
                        </Typography>
                        <Typography variant="body1">
                          {selectedPattern.details.success_rate}%
                        </Typography>
                      </Grid>
                    </Grid>
                    
                    <Divider sx={{ my: 2 }} />
                    
                    <Typography variant="subtitle1" gutterBottom>
                      Pattern Details:
                    </Typography>
                    
                    <List dense>
                      {Object.entries(selectedPattern.details).map(([key, value]) => (
                        key !== 'success_rate' && (
                          <ListItem key={key} sx={{ py: 0 }}>
                            <ListItemText 
                              primary={
                                <span>
                                  <Typography component="span" color="textSecondary" sx={{ textTransform: 'capitalize' }}>
                                    {key.replace(/_/g, ' ')}:
                                  </Typography>{' '}
                                  <Typography component="span">
                                    {Array.isArray(value) ? value.join(', ') : value.toString()}
                                  </Typography>
                                </span>
                              }
                            />
                          </ListItem>
                        )
                      ))}
                    </List>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
                      <Button 
                        variant="contained" 
                        color="primary" 
                        startIcon={<LightbulbIcon />}
                        onClick={handleCreateAutomation}
                      >
                        Create Automation
                      </Button>
                    </Box>
                  </>
                ) : (
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', p: 3 }}>
                    <TrendingUpIcon sx={{ fontSize: 60, color: 'text.disabled', mb: 2 }} />
                    <Typography variant="h6" color="textSecondary" align="center">
                      Select a pattern to view details
                    </Typography>
                  </Box>
                )}
              </Paper>
            </Grid>
          </Grid>
        </>
      )}
    </Box>
  );
}

export default LearningProgress;
