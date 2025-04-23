import React, { useState, useEffect, useContext } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Button,
  Divider,
  IconButton,
  Tooltip
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import StorageIcon from '@mui/icons-material/Storage';
import FilterListIcon from '@mui/icons-material/FilterList';
import VisibilityIcon from '@mui/icons-material/Visibility';
import PrivacyTipIcon from '@mui/icons-material/PrivacyTip';
import InfoIcon from '@mui/icons-material/Info';
import DateRangeIcon from '@mui/icons-material/DateRange';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { LocalizationProvider, DatePicker } from '@mui/x-date-pickers';
import AlertContext from '../context/AlertContext';

function ActivityMonitor() {
  const { showAlert } = useContext(AlertContext);
  const [loading, setLoading] = useState(true);
  const [activities, setActivities] = useState([]);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [activityType, setActivityType] = useState('all');
  const [application, setApplication] = useState('all');
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [appList, setAppList] = useState([]);
  const [showFilters, setShowFilters] = useState(false);

  useEffect(() => {
    // Simulate loading activities from API
    const timer = setTimeout(() => {
      const mockActivities = generateMockActivities(100);
      setActivities(mockActivities);
      
      // Extract unique app names for filter
      const apps = [...new Set(mockActivities.map(activity => activity.application))];
      setAppList(apps);
      
      setLoading(false);
    }, 1500);
    
    return () => clearTimeout(timer);
  }, []);

  // Generate mock activity data
  const generateMockActivities = (count) => {
    const activityTypes = ['keyboard', 'mouse', 'system'];
    const apps = ['Word', 'Excel', 'Chrome', 'Outlook', 'Visual Studio', 'Slack', 'Teams', 'File Explorer'];
    const actions = {
      keyboard: ['key_press', 'shortcut', 'text_input'],
      mouse: ['click', 'scroll', 'move'],
      system: ['application_focus', 'file_open', 'system_idle', 'notification']
    };
    
    const result = [];
    const now = new Date();
    
    for (let i = 0; i < count; i++) {
      const type = activityTypes[Math.floor(Math.random() * activityTypes.length)];
      const app = apps[Math.floor(Math.random() * apps.length)];
      const action = actions[type][Math.floor(Math.random() * actions[type].length)];
      
      // Random time in the past week
      const timestamp = new Date(now - Math.floor(Math.random() * 7 * 24 * 60 * 60 * 1000));
      
      result.push({
        id: i + 1,
        activity_type: type,
        application: app,
        action: action,
        timestamp: timestamp.toISOString(),
        metadata: {
          duration: Math.floor(Math.random() * 120),
          position: type === 'mouse' ? { x: Math.floor(Math.random() * 1920), y: Math.floor(Math.random() * 1080) } : null
        }
      });
    }
    
    // Sort by timestamp (newest first)
    return result.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  };

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleActivityTypeChange = (event) => {
    setActivityType(event.target.value);
    setPage(0);
  };

  const handleApplicationChange = (event) => {
    setApplication(event.target.value);
    setPage(0);
  };

  const handleApplyFilters = () => {
    setLoading(true);
    
    // Simulate API call with filters
    setTimeout(() => {
      // In a real app, this would be an API call with the filter parameters
      const filtered = generateMockActivities(100).filter(activity => {
        if (activityType !== 'all' && activity.activity_type !== activityType) return false;
        if (application !== 'all' && activity.application !== application) return false;
        if (startDate && new Date(activity.timestamp) < startDate) return false;
        if (endDate && new Date(activity.timestamp) > endDate) return false;
        return true;
      });
      
      setActivities(filtered);
      setLoading(false);
      showAlert('Filters applied', 'success');
    }, 1000);
  };

  const handleResetFilters = () => {
    setActivityType('all');
    setApplication('all');
    setStartDate(null);
    setEndDate(null);
    
    handleApplyFilters();
  };

  const handleDeleteActivity = (id) => {
    // In a real app, this would call an API to delete the activity
    setActivities(activities.filter(activity => activity.id !== id));
    showAlert('Activity deleted', 'success');
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const getActivityTypeColor = (type) => {
    switch (type) {
      case 'keyboard': return 'primary';
      case 'mouse': return 'success';
      case 'system': return 'info';
      default: return 'default';
    }
  };

  // Filter activities based on current filters
  const filteredActivities = activities;

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Activity Monitor
      </Typography>
      
      <Paper sx={{ p: 2, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            Activity Records
            <Tooltip title="Activities are monitored to learn your behavior patterns">
              <IconButton size="small" sx={{ ml: 1 }}>
                <InfoIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Typography>
          <Box>
            <Tooltip title="Show/Hide Filters">
              <IconButton onClick={() => setShowFilters(!showFilters)}>
                <FilterListIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Privacy Information">
              <IconButton>
                <PrivacyTipIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
        
        {showFilters && (
          <Box sx={{ mb: 2 }}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} sm={3}>
                <FormControl fullWidth size="small">
                  <InputLabel>Activity Type</InputLabel>
                  <Select
                    value={activityType}
                    label="Activity Type"
                    onChange={handleActivityTypeChange}
                  >
                    <MenuItem value="all">All Types</MenuItem>
                    <MenuItem value="keyboard">Keyboard</MenuItem>
                    <MenuItem value="mouse">Mouse</MenuItem>
                    <MenuItem value="system">System</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} sm={3}>
                <FormControl fullWidth size="small">
                  <InputLabel>Application</InputLabel>
                  <Select
                    value={application}
                    label="Application"
                    onChange={handleApplicationChange}
                  >
                    <MenuItem value="all">All Applications</MenuItem>
                    {appList.map(app => (
                      <MenuItem key={app} value={app}>{app}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} sm={3}>
                <LocalizationProvider dateAdapter={AdapterDateFns}>
                  <DatePicker
                    label="Start Date"
                    value={startDate}
                    onChange={setStartDate}
                    renderInput={(params) => <TextField {...params} size="small" fullWidth />}
                  />
                </LocalizationProvider>
              </Grid>
              
              <Grid item xs={12} sm={3}>
                <LocalizationProvider dateAdapter={AdapterDateFns}>
                  <DatePicker
                    label="End Date"
                    value={endDate}
                    onChange={setEndDate}
                    renderInput={(params) => <TextField {...params} size="small" fullWidth />}
                  />
                </LocalizationProvider>
              </Grid>
            </Grid>
            
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
              <Button onClick={handleResetFilters} sx={{ mr: 1 }}>
                Reset
              </Button>
              <Button variant="contained" onClick={handleApplyFilters}>
                Apply Filters
              </Button>
            </Box>
            
            <Divider sx={{ my: 2 }} />
          </Box>
        )}
        
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        ) : (
          <>
            <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
              {filteredActivities.length} activities found. Raw activity data is automatically deleted after 24 hours.
            </Typography>
            
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Type</TableCell>
                    <TableCell>Application</TableCell>
                    <TableCell>Action</TableCell>
                    <TableCell>Timestamp</TableCell>
                    <TableCell>Details</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredActivities
                    .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                    .map((activity) => (
                      <TableRow key={activity.id}>
                        <TableCell>
                          <Chip 
                            label={activity.activity_type} 
                            size="small"
                            color={getActivityTypeColor(activity.activity_type)}
                          />
                        </TableCell>
                        <TableCell>{activity.application}</TableCell>
                        <TableCell>{activity.action}</TableCell>
                        <TableCell>{formatTimestamp(activity.timestamp)}</TableCell>
                        <TableCell>
                          {activity.activity_type === 'mouse' && activity.metadata?.position && 
                            `Position: (${activity.metadata.position.x}, ${activity.metadata.position.y})`
                          }
                          {activity.activity_type === 'system' && 
                            `Duration: ${activity.metadata?.duration || 0}s`
                          }
                        </TableCell>
                        <TableCell align="right">
                          <Tooltip title="View Details">
                            <IconButton size="small">
                              <VisibilityIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton 
                              size="small" 
                              color="error"
                              onClick={() => handleDeleteActivity(activity.id)}
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </TableContainer>
            
            <TablePagination
              rowsPerPageOptions={[5, 10, 25, 50]}
              component="div"
              count={filteredActivities.length}
              rowsPerPage={rowsPerPage}
              page={page}
              onPageChange={handleChangePage}
              onRowsPerPageChange={handleChangeRowsPerPage}
            />
          </>
        )}
      </Paper>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Processing Status
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
              <CircularProgress variant="determinate" value={75} size={60} thickness={5} sx={{ mr: 2 }} />
              <Box>
                <Typography variant="h5">75%</Typography>
                <Typography variant="body2" color="textSecondary">
                  Activities processed
                </Typography>
              </Box>
            </Box>
            <Typography variant="body2" sx={{ mt: 2 }}>
              Raw activity data is automatically deleted after 24 hours once processed by the learning engine.
              This ensures your privacy while still allowing the assistant to learn from your behavior.
            </Typography>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Data Storage
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
              <Box sx={{ position: 'relative', display: 'inline-flex', mr: 2 }}>
                <CircularProgress variant="determinate" value={30} size={60} thickness={5} color="secondary" />
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
                  <StorageIcon color="secondary" />
                </Box>
              </Box>
              <Box>
                <Typography variant="h5">120 MB</Typography>
                <Typography variant="body2" color="textSecondary">
                  Total data storage
                </Typography>
              </Box>
            </Box>
            <Typography variant="body2" sx={{ mt: 2 }}>
              All data is stored locally on your device. No activity data is sent to external servers.
              Your privacy is protected through local processing and storage.
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default ActivityMonitor;
