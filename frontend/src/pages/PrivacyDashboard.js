import React, { useState, useEffect, useContext } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Switch,
  FormGroup,
  FormControlLabel,
  Divider,
  Button,
  Card,
  CardContent,
  CardActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Slider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  CircularProgress,
  IconButton,
  Tooltip
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import DeleteIcon from '@mui/icons-material/Delete';
import DownloadIcon from '@mui/icons-material/Download';
import SecurityIcon from '@mui/icons-material/Security';
import SettingsIcon from '@mui/icons-material/Settings';
import InfoIcon from '@mui/icons-material/Info';
import HistoryIcon from '@mui/icons-material/History';
import AlertContext from '../context/AlertContext';
import axios from 'axios';

// API URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function PrivacyDashboard() {
  const { showAlert } = useContext(AlertContext);
  const [loading, setLoading] = useState(true);
  const [userId, setUserId] = useState(1); // This would normally come from authentication
  const [privacySettings, setPrivacySettings] = useState({});
  const [retentionPolicies, setRetentionPolicies] = useState([]);
  const [accessLogs, setAccessLogs] = useState([]);
  const [exportDialog, setExportDialog] = useState(false);
  const [deleteDialog, setDeleteDialog] = useState(false);
  const [selectedDataTypes, setSelectedDataTypes] = useState([]);
  const [dataTypeOptions] = useState([
    { id: 'activity_logs', label: 'Activity Logs' },
    { id: 'keyboard_data', label: 'Keyboard Data' },
    { id: 'mouse_data', label: 'Mouse Data' },
    { id: 'application_usage', label: 'Application Usage' },
    { id: 'behavior_models', label: 'Behavior Models' },
    { id: 'automation_recipes', label: 'Automation Recipes' }
  ]);

  // Fetch privacy settings on component mount
  useEffect(() => {
    const fetchPrivacyData = async () => {
      setLoading(true);
      try {
        // Fetch privacy settings
        const settingsResponse = await axios.get(`${API_BASE_URL}/privacy/settings/${userId}`);
        setPrivacySettings(settingsResponse.data);

        // Fetch retention policies
        const policiesResponse = await axios.get(`${API_BASE_URL}/privacy/retention_policies/${userId}`);
        setRetentionPolicies(policiesResponse.data);

        // Fetch access logs
        const logsResponse = await axios.get(`${API_BASE_URL}/privacy/access_logs/${userId}`);
        setAccessLogs(logsResponse.data);

        setLoading(false);
      } catch (error) {
        console.error('Error fetching privacy data:', error);
        showAlert('error', 'Failed to load privacy data. Please try again later.');
        setLoading(false);
      }
    };

    fetchPrivacyData();
  }, [userId, showAlert]);

  // Handle privacy settings changes
  const handleSettingChange = async (event) => {
    const { name, checked } = event.target;
    const updatedSettings = { ...privacySettings, [name]: checked };
    
    setPrivacySettings(updatedSettings);
    
    try {
      await axios.put(`${API_BASE_URL}/privacy/settings/${userId}`, {
        [name]: checked
      });
      showAlert('success', 'Privacy setting updated successfully');
    } catch (error) {
      console.error('Error updating privacy setting:', error);
      showAlert('error', 'Failed to update privacy setting');
      // Revert the change
      setPrivacySettings(privacySettings);
    }
  };

  // Handle retention policy changes
  const handleRetentionChange = async (policyId, field, value) => {
    const updatedPolicies = retentionPolicies.map(policy => {
      if (policy.id === policyId) {
        return { ...policy, [field]: value };
      }
      return policy;
    });
    
    setRetentionPolicies(updatedPolicies);
    
    try {
      await axios.put(`${API_BASE_URL}/privacy/retention_policies/${userId}`, {
        policy_id: policyId,
        [field]: value
      });
      showAlert('success', 'Retention policy updated successfully');
    } catch (error) {
      console.error('Error updating retention policy:', error);
      showAlert('error', 'Failed to update retention policy');
      // Fetch the most recent data
      const response = await axios.get(`${API_BASE_URL}/privacy/retention_policies/${userId}`);
      setRetentionPolicies(response.data);
    }
  };

  // Handle data export
  const handleExportData = async () => {
    if (selectedDataTypes.length === 0) {
      showAlert('warning', 'Please select at least one data type to export');
      return;
    }

    try {
      const response = await axios.post(`${API_BASE_URL}/privacy/request_data_export/${userId}`, {
        data_types: selectedDataTypes
      });
      
      showAlert('success', 'Data export request submitted successfully');
      setExportDialog(false);
      setSelectedDataTypes([]);
    } catch (error) {
      console.error('Error requesting data export:', error);
      showAlert('error', 'Failed to request data export');
    }
  };

  // Handle data deletion
  const handleDeleteData = async () => {
    if (selectedDataTypes.length === 0) {
      showAlert('warning', 'Please select at least one data type to delete');
      return;
    }

    try {
      const response = await axios.post(`${API_BASE_URL}/privacy/request_data_deletion/${userId}`, {
        data_types: selectedDataTypes
      });
      
      showAlert('success', 'Data deletion request submitted successfully');
      setDeleteDialog(false);
      setSelectedDataTypes([]);
    } catch (error) {
      console.error('Error requesting data deletion:', error);
      showAlert('error', 'Failed to request data deletion');
    }
  };

  // Toggle data type selection
  const toggleDataType = (dataTypeId) => {
    setSelectedDataTypes(prevSelected => {
      if (prevSelected.includes(dataTypeId)) {
        return prevSelected.filter(id => id !== dataTypeId);
      } else {
        return [...prevSelected, dataTypeId];
      }
    });
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ padding: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
        <SecurityIcon fontSize="large" sx={{ mr: 1 }} /> 
        Privacy Dashboard
      </Typography>
      <Typography variant="subtitle1" color="text.secondary" paragraph>
        Control how your data is collected, used, and stored by the AI Assistant.
      </Typography>

      <Grid container spacing={3}>
        {/* Privacy Settings Panel */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <SettingsIcon sx={{ mr: 1 }} /> 
              Data Collection Settings
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Configure which data the AI Assistant can collect and analyze.
            </Typography>
            <Divider sx={{ my: 2 }} />
            
            <FormGroup>
              <FormControlLabel
                control={<Switch 
                  checked={privacySettings.data_collection_enabled || false}
                  onChange={handleSettingChange}
                  name="data_collection_enabled"
                />}
                label="Enable Data Collection (Master Switch)"
              />
              
              <Box sx={{ ml: 3, mt: 2 }}>
                <FormControlLabel
                  control={<Switch 
                    checked={privacySettings.keyboard_monitoring || false}
                    onChange={handleSettingChange}
                    name="keyboard_monitoring"
                    disabled={!privacySettings.data_collection_enabled}
                  />}
                  label="Keyboard Activity"
                />
                <FormControlLabel
                  control={<Switch 
                    checked={privacySettings.mouse_monitoring || false}
                    onChange={handleSettingChange}
                    name="mouse_monitoring"
                    disabled={!privacySettings.data_collection_enabled}
                  />}
                  label="Mouse Activity"
                />
                <FormControlLabel
                  control={<Switch 
                    checked={privacySettings.screen_monitoring || false}
                    onChange={handleSettingChange}
                    name="screen_monitoring"
                    disabled={!privacySettings.data_collection_enabled}
                  />}
                  label="Screen Content"
                />
                <FormControlLabel
                  control={<Switch 
                    checked={privacySettings.application_monitoring || false}
                    onChange={handleSettingChange}
                    name="application_monitoring"
                    disabled={!privacySettings.data_collection_enabled}
                  />}
                  label="Application Usage"
                />
                <FormControlLabel
                  control={<Switch 
                    checked={privacySettings.browser_history_monitoring || false}
                    onChange={handleSettingChange}
                    name="browser_history_monitoring"
                    disabled={!privacySettings.data_collection_enabled}
                  />}
                  label="Browser History"
                />
                <FormControlLabel
                  control={<Switch 
                    checked={privacySettings.email_monitoring || false}
                    onChange={handleSettingChange}
                    name="email_monitoring"
                    disabled={!privacySettings.data_collection_enabled}
                  />}
                  label="Email Contents"
                />
                <FormControlLabel
                  control={<Switch 
                    checked={privacySettings.calendar_monitoring || false}
                    onChange={handleSettingChange}
                    name="calendar_monitoring"
                    disabled={!privacySettings.data_collection_enabled}
                  />}
                  label="Calendar Events"
                />
                <FormControlLabel
                  control={<Switch 
                    checked={privacySettings.file_monitoring || false}
                    onChange={handleSettingChange}
                    name="file_monitoring"
                    disabled={!privacySettings.data_collection_enabled}
                  />}
                  label="File Access"
                />
              </Box>
            </FormGroup>
            
            <Divider sx={{ my: 2 }} />
            
            <FormGroup>
              <FormControlLabel
                control={<Switch 
                  checked={privacySettings.anonymize_sensitive_data || false}
                  onChange={handleSettingChange}
                  name="anonymize_sensitive_data"
                />}
                label="Anonymize Sensitive Data"
              />
              <FormControlLabel
                control={<Switch 
                  checked={privacySettings.share_data_with_third_parties || false}
                  onChange={handleSettingChange}
                  name="share_data_with_third_parties"
                />}
                label="Allow Sharing with Third Parties"
              />
            </FormGroup>
          </Paper>
        </Grid>

        {/* Data Retention Panel */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <HistoryIcon sx={{ mr: 1 }} /> 
              Data Retention Policies
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Control how long your data is stored before being anonymized or deleted.
            </Typography>
            <Divider sx={{ my: 2 }} />

            {retentionPolicies.map((policy) => (
              <Accordion key={policy.id} sx={{ mb: 1 }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>{policy.data_type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <Typography variant="body2" gutterBottom>
                        Keep data for {policy.retention_days} days
                      </Typography>
                      <Slider
                        value={policy.retention_days}
                        min={1}
                        max={365}
                        step={1}
                        valueLabelDisplay="auto"
                        onChange={(_, value) => handleRetentionChange(policy.id, 'retention_days', value)}
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <Typography variant="body2" gutterBottom>
                        Anonymize after {policy.anonymize_after_days || 'never'} days
                      </Typography>
                      <Slider
                        value={policy.anonymize_after_days || 365}
                        min={1}
                        max={365}
                        step={1}
                        valueLabelDisplay="auto"
                        onChange={(_, value) => handleRetentionChange(policy.id, 'anonymize_after_days', value)}
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <Typography variant="body2" gutterBottom>
                        Delete after {policy.delete_after_days || 'never'} days
                      </Typography>
                      <Slider
                        value={policy.delete_after_days || 365}
                        min={1}
                        max={365}
                        step={1}
                        valueLabelDisplay="auto"
                        onChange={(_, value) => handleRetentionChange(policy.id, 'delete_after_days', value)}
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            ))}

            <Box sx={{ mt: 3, display: 'flex', justifyContent: 'space-between' }}>
              <Button
                variant="outlined"
                startIcon={<DownloadIcon />}
                onClick={() => setExportDialog(true)}
              >
                Export My Data
              </Button>
              <Button
                variant="outlined"
                color="error"
                startIcon={<DeleteIcon />}
                onClick={() => setDeleteDialog(true)}
              >
                Delete My Data
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* Access Logs Panel */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <InfoIcon sx={{ mr: 1 }} /> 
              Data Access Logs
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              View records of when your data was accessed or modified.
            </Typography>
            <Divider sx={{ my: 2 }} />

            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Timestamp</TableCell>
                    <TableCell>Operation</TableCell>
                    <TableCell>Accessed By</TableCell>
                    <TableCell>Details</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {accessLogs.length > 0 ? (
                    accessLogs.map((log) => (
                      <TableRow key={log.id}>
                        <TableCell>{new Date(log.timestamp).toLocaleString()}</TableCell>
                        <TableCell>
                          <Chip 
                            label={log.operation.replace('_', ' ')} 
                            color={log.operation.includes('delete') ? 'error' : 'default'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>{log.accessed_by === userId ? 'You' : log.accessed_by}</TableCell>
                        <TableCell>{log.details}</TableCell>
                      </TableRow>
                    ))
                  ) : (
                    <TableRow>
                      <TableCell colSpan={4} align="center">No access logs found</TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* Data Export Dialog */}
      <Dialog open={exportDialog} onClose={() => setExportDialog(false)}>
        <DialogTitle>Export My Data</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Select which types of data you would like to export. The export will be prepared and a download link will be sent to your email.
          </DialogContentText>
          <Box sx={{ mt: 2 }}>
            {dataTypeOptions.map((option) => (
              <FormControlLabel
                key={option.id}
                control={
                  <Switch
                    checked={selectedDataTypes.includes(option.id)}
                    onChange={() => toggleDataType(option.id)}
                  />
                }
                label={option.label}
              />
            ))}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setExportDialog(false)}>Cancel</Button>
          <Button 
            variant="contained" 
            onClick={handleExportData}
            disabled={selectedDataTypes.length === 0}
          >
            Export
          </Button>
        </DialogActions>
      </Dialog>

      {/* Data Deletion Dialog */}
      <Dialog open={deleteDialog} onClose={() => setDeleteDialog(false)}>
        <DialogTitle>Delete My Data</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Select which types of data you would like to delete. This action cannot be undone, and it may affect the AI Assistant's ability to provide personalized services.
          </DialogContentText>
          <Box sx={{ mt: 2 }}>
            {dataTypeOptions.map((option) => (
              <FormControlLabel
                key={option.id}
                control={
                  <Switch
                    checked={selectedDataTypes.includes(option.id)}
                    onChange={() => toggleDataType(option.id)}
                  />
                }
                label={option.label}
              />
            ))}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog(false)}>Cancel</Button>
          <Button 
            variant="contained" 
            color="error" 
            onClick={handleDeleteData}
            disabled={selectedDataTypes.length === 0}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default PrivacyDashboard;
