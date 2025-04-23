import React, { useState, useContext, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Divider,
  Switch,
  FormControlLabel,
  Slider,
  Button,
  Grid,
  Card,
  CardContent,
  Alert,
  TextField,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  CircularProgress
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SecurityIcon from '@mui/icons-material/Security';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import DataUsageIcon from '@mui/icons-material/DataUsage';
import StorageIcon from '@mui/icons-material/Storage';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import BlockIcon from '@mui/icons-material/Block';
import SettingsBackupRestoreIcon from '@mui/icons-material/SettingsBackupRestore';
import AuthContext from '../context/AuthContext';
import AlertContext from '../context/AlertContext';

function Settings() {
  const { user } = useContext(AuthContext);
  const { showAlert } = useContext(AlertContext);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  // Settings state
  const [settings, setSettings] = useState({
    monitoringEnabled: true,
    learningEnabled: true,
    automationEnabled: false,
    privacyLevel: 2,
    dataRetentionDays: 30,
    blockedApplications: ['Banking App', 'Password Manager'],
    confirmationRequired: true,
    startOnSystemBoot: true,
    notificationsEnabled: true,
    backupFrequency: 'weekly',
    customModels: []
  });

  useEffect(() => {
    // Simulate loading settings from API
    const timer = setTimeout(() => {
      // In a real app, this would be an API call to load user settings
      setLoading(false);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, []);

  const handleToggleChange = (event) => {
    setSettings({
      ...settings,
      [event.target.name]: event.target.checked
    });
  };

  const handleSliderChange = (name) => (event, newValue) => {
    setSettings({
      ...settings,
      [name]: newValue
    });
  };

  const handleInputChange = (name) => (event) => {
    setSettings({
      ...settings,
      [name]: event.target.value
    });
  };

  const handleBlockApp = () => {
    // This would open a dialog to select an app to block
    // For now, we'll just simulate adding a blocked app
    const newApp = 'New Blocked App';
    if (!settings.blockedApplications.includes(newApp)) {
      setSettings({
        ...settings,
        blockedApplications: [...settings.blockedApplications, newApp]
      });
      showAlert(`Added ${newApp} to blocked applications`, 'success');
    }
  };

  const handleUnblockApp = (app) => {
    setSettings({
      ...settings,
      blockedApplications: settings.blockedApplications.filter(a => a !== app)
    });
    showAlert(`Removed ${app} from blocked applications`, 'success');
  };

  const handleSaveSettings = async () => {
    setSaving(true);
    
    try {
      // Simulate API call to save settings
      await new Promise(resolve => setTimeout(resolve, 1000));
      showAlert('Settings saved successfully', 'success');
    } catch (error) {
      showAlert('Failed to save settings', 'error');
    } finally {
      setSaving(false);
    }
  };

  const handleResetSettings = () => {
    // Reset to default settings
    setSettings({
      monitoringEnabled: true,
      learningEnabled: true,
      automationEnabled: false,
      privacyLevel: 2,
      dataRetentionDays: 30,
      blockedApplications: [],
      confirmationRequired: true,
      startOnSystemBoot: true,
      notificationsEnabled: true,
      backupFrequency: 'weekly',
      customModels: []
    });
    showAlert('Settings reset to defaults', 'info');
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Settings
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        These settings control how your AI Assistant monitors, learns, and automates tasks on your device.
      </Alert>
      
      <Grid container spacing={3}>
        {/* Core Features */}
        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Core Features
              </Typography>
              <Divider sx={{ mb: 2 }} />
              
              <List>
                <ListItem>
                  <ListItemIcon>
                    <DataUsageIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Activity Monitoring" 
                    secondary="Allow assistant to monitor your activities"
                  />
                  <ListItemSecondaryAction>
                    <Switch
                      edge="end"
                      name="monitoringEnabled"
                      checked={settings.monitoringEnabled}
                      onChange={handleToggleChange}
                    />
                  </ListItemSecondaryAction>
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <AutoAwesomeIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Learning Engine" 
                    secondary="Allow assistant to learn from your behavior"
                  />
                  <ListItemSecondaryAction>
                    <Switch
                      edge="end"
                      name="learningEnabled"
                      checked={settings.learningEnabled}
                      onChange={handleToggleChange}
                      disabled={!settings.monitoringEnabled}
                    />
                  </ListItemSecondaryAction>
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <AutoAwesomeIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Task Automation" 
                    secondary="Allow assistant to automate tasks for you"
                  />
                  <ListItemSecondaryAction>
                    <Switch
                      edge="end"
                      name="automationEnabled"
                      checked={settings.automationEnabled}
                      onChange={handleToggleChange}
                      disabled={!settings.learningEnabled}
                    />
                  </ListItemSecondaryAction>
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <AutoAwesomeIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Require Confirmation" 
                    secondary="Require confirmation before executing automated tasks"
                  />
                  <ListItemSecondaryAction>
                    <Switch
                      edge="end"
                      name="confirmationRequired"
                      checked={settings.confirmationRequired}
                      onChange={handleToggleChange}
                      disabled={!settings.automationEnabled}
                    />
                  </ListItemSecondaryAction>
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <StorageIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Start on System Boot" 
                    secondary="Start the assistant when your computer starts"
                  />
                  <ListItemSecondaryAction>
                    <Switch
                      edge="end"
                      name="startOnSystemBoot"
                      checked={settings.startOnSystemBoot}
                      onChange={handleToggleChange}
                    />
                  </ListItemSecondaryAction>
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Privacy & Security */}
        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Privacy & Security
              </Typography>
              <Divider sx={{ mb: 2 }} />
              
              <Typography gutterBottom>
                Privacy Level
              </Typography>
              <Box sx={{ px: 2, pb: 2 }}>
                <Slider
                  value={settings.privacyLevel}
                  min={1}
                  max={3}
                  step={1}
                  marks={[
                    { value: 1, label: 'Basic' },
                    { value: 2, label: 'Balanced' },
                    { value: 3, label: 'Maximum' }
                  ]}
                  onChange={handleSliderChange('privacyLevel')}
                  valueLabelDisplay="off"
                />
              </Box>
              
              <Typography variant="body2" color="textSecondary" paragraph>
                {settings.privacyLevel === 1 && 'Basic: Collect all activity data for best personalization'}
                {settings.privacyLevel === 2 && 'Balanced: Collect activity data but anonymize sensitive information'}
                {settings.privacyLevel === 3 && 'Maximum: Minimal data collection, maximum privacy protection'}
              </Typography>
              
              <Typography gutterBottom sx={{ mt: 2 }}>
                Data Retention Period (days)
              </Typography>
              <Box sx={{ px: 2, pb: 2 }}>
                <Slider
                  value={settings.dataRetentionDays}
                  min={1}
                  max={90}
                  step={1}
                  onChange={handleSliderChange('dataRetentionDays')}
                  valueLabelDisplay="on"
                  disabled={settings.privacyLevel === 3}
                />
              </Box>
              
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Raw activity data will be deleted after {settings.dataRetentionDays} days.
                Learned patterns will be retained.
              </Typography>
              
              <Accordion sx={{ mt: 2 }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <BlockIcon sx={{ mr: 1 }} />
                    <Typography>Blocked Applications ({settings.blockedApplications.length})</Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Typography variant="body2" color="textSecondary" paragraph>
                    The assistant will not monitor or automate these applications.
                  </Typography>
                  
                  <List dense>
                    {settings.blockedApplications.map((app) => (
                      <ListItem key={app}>
                        <ListItemIcon>
                          <VisibilityOffIcon fontSize="small" />
                        </ListItemIcon>
                        <ListItemText primary={app} />
                        <ListItemSecondaryAction>
                          <Button 
                            size="small" 
                            color="error" 
                            onClick={() => handleUnblockApp(app)}
                          >
                            Remove
                          </Button>
                        </ListItemSecondaryAction>
                      </ListItem>
                    ))}
                    
                    {settings.blockedApplications.length === 0 && (
                      <ListItem>
                        <ListItemText primary="No blocked applications" />
                      </ListItem>
                    )}
                  </List>
                  
                  <Button 
                    variant="outlined" 
                    size="small" 
                    sx={{ mt: 1 }}
                    onClick={handleBlockApp}
                  >
                    Block Application
                  </Button>
                </AccordionDetails>
              </Accordion>
            </CardContent>
          </Card>
        </Grid>

        {/* System & Backup */}
        <Grid item xs={12}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System & Backup
              </Typography>
              <Divider sx={{ mb: 2 }} />
              
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    select
                    label="Backup Frequency"
                    value={settings.backupFrequency}
                    onChange={handleInputChange('backupFrequency')}
                    fullWidth
                    SelectProps={{
                      native: true,
                    }}
                  >
                    <option value="daily">Daily</option>
                    <option value="weekly">Weekly</option>
                    <option value="monthly">Monthly</option>
                    <option value="never">Never</option>
                  </TextField>
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Backup Location"
                    value="C:\\Users\\AppData\\Local\\AI-Assistant\\backups"
                    fullWidth
                    disabled
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
                    <Button 
                      variant="outlined" 
                      startIcon={<SettingsBackupRestoreIcon />}
                    >
                      Backup Now
                    </Button>
                    <Button 
                      variant="outlined" 
                      color="secondary"
                    >
                      Restore from Backup
                    </Button>
                    <Button 
                      variant="outlined" 
                      color="error" 
                      startIcon={<StorageIcon />}
                    >
                      Clear Learned Data
                    </Button>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
        <Button 
          variant="outlined" 
          color="error" 
          onClick={handleResetSettings}
        >
          Reset to Defaults
        </Button>
        <Button 
          variant="contained" 
          color="primary" 
          onClick={handleSaveSettings}
          disabled={saving}
        >
          {saving ? <CircularProgress size={24} /> : 'Save Settings'}
        </Button>
      </Box>
    </Box>
  );
}

export default Settings;
