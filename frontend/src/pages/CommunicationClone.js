import React, { useState, useEffect, useContext } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Switch,
  FormGroup,
  FormControlLabel,
  Divider,
  TextField,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Slider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  AlertTitle,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Radio,
  RadioGroup,
  FormControl,
  FormLabel
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChatIcon from '@mui/icons-material/Chat';
import InsightsIcon from '@mui/icons-material/Insights';
import SettingsIcon from '@mui/icons-material/Settings';
import TextsmsIcon from '@mui/icons-material/Textsms';
import EmailIcon from '@mui/icons-material/Email';
import InfoIcon from '@mui/icons-material/Info';
import SecurityIcon from '@mui/icons-material/Security';
import SampleTextIcon from '@mui/icons-material/Description';
import SendIcon from '@mui/icons-material/Send';
import AlertContext from '../context/AlertContext';
import axios from 'axios';

// API URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function CommunicationClone() {
  const { showAlert } = useContext(AlertContext);
  const [loading, setLoading] = useState(true);
  const [userId, setUserId] = useState(1); // This would normally come from authentication
  const [settings, setSettings] = useState({
    enabled: false,
    consent_given: false,
    style_mimicry_level: 0.7,
    preferred_model: 'local',
    collect_training_data: true
  });
  
  const [styleProfile, setStyleProfile] = useState({
    complexity_score: 0.5,
    formality_score: 0.5,
    emoji_usage_rate: 0,
    avg_sentence_length: 15,
    avg_word_length: 4.5,
    favorite_phrases: [],
    word_frequencies: {},
    summary: {
      writing_style: 'casual',
      complexity: 'simple',
      emoji_usage: 'rare',
      top_words: [],
      sample_phrases: [],
      mimicry_confidence: 0.1
    }
  });
  
  const [consentDialogOpen, setConsentDialogOpen] = useState(false);
  const [generationDialogOpen, setGenerationDialogOpen] = useState(false);
  const [generationContext, setGenerationContext] = useState('email_draft');
  const [prompt, setPrompt] = useState('');
  const [generatedText, setGeneratedText] = useState('');
  const [generationLoading, setGenerationLoading] = useState(false);

  // Fetch communication clone settings and style profile on component mount
  useEffect(() => {
    const fetchCommunicationData = async () => {
      setLoading(true);
      try {
        // Fetch communication settings
        const settingsResponse = await axios.get(`${API_BASE_URL}/communication/settings/${userId}`);
        setSettings(settingsResponse.data);

        // Fetch style profile
        const profileResponse = await axios.get(`${API_BASE_URL}/communication/style_profile/${userId}`);
        setStyleProfile(profileResponse.data);

        setLoading(false);
      } catch (error) {
        console.error('Error fetching communication data:', error);
        showAlert('error', 'Failed to load communication profile data. Please try again later.');
        setLoading(false);
      }
    };

    fetchCommunicationData();
  }, [userId, showAlert]);

  // Handle settings changes
  const handleSettingChange = async (event) => {
    const { name, checked, value } = event.target;
    const newValue = name === 'style_mimicry_level' ? parseFloat(value) : checked;
    
    // Special handling for enabling the feature
    if (name === 'enabled' && checked && !settings.consent_given) {
      setConsentDialogOpen(true);
      return;
    }
    
    const updatedSettings = { ...settings, [name]: newValue };
    setSettings(updatedSettings);
    
    try {
      await axios.put(`${API_BASE_URL}/communication/settings/${userId}`, {
        [name]: newValue
      });
      showAlert('success', 'Communication setting updated successfully');
    } catch (error) {
      console.error('Error updating communication setting:', error);
      showAlert('error', 'Failed to update communication setting');
      // Revert the change
      setSettings(settings);
    }
  };

  // Handle model preference change
  const handleModelChange = async (event) => {
    const { value } = event.target;
    
    const updatedSettings = { ...settings, preferred_model: value };
    setSettings(updatedSettings);
    
    try {
      await axios.put(`${API_BASE_URL}/communication/settings/${userId}`, {
        preferred_model: value
      });
      showAlert('success', 'Model preference updated successfully');
    } catch (error) {
      console.error('Error updating model preference:', error);
      showAlert('error', 'Failed to update model preference');
      // Revert the change
      setSettings(settings);
    }
  };

  // Handle consent dialog
  const handleConsentConfirm = async () => {
    // Update both consent and enabled settings
    const updatedSettings = { ...settings, consent_given: true, enabled: true };
    setSettings(updatedSettings);
    
    try {
      await axios.put(`${API_BASE_URL}/communication/settings/${userId}`, {
        consent_given: true,
        enabled: true
      });
      showAlert('success', 'Communication cloning enabled');
      setConsentDialogOpen(false);
    } catch (error) {
      console.error('Error updating consent settings:', error);
      showAlert('error', 'Failed to update consent settings');
      // Revert the changes
      setSettings(settings);
    }
  };

  // Handle text generation
  const handleGenerateText = async () => {
    if (!prompt) {
      showAlert('warning', 'Please enter a prompt to start generation');
      return;
    }

    setGenerationLoading(true);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/communication/generate/${userId}`, {
        prompt,
        context: generationContext,
        max_length: 500
      });
      
      setGeneratedText(response.data.text);
      setGenerationLoading(false);
    } catch (error) {
      console.error('Error generating text:', error);
      showAlert('error', 'Failed to generate text');
      setGenerationLoading(false);
    }
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
        <ChatIcon fontSize="large" sx={{ mr: 1 }} /> 
        Communication Clone
      </Typography>
      <Typography variant="subtitle1" color="text.secondary" paragraph>
        Analyze and clone your communication style for email drafts, messages, and notes.
      </Typography>

      {/* Consent Alert */}
      {!settings.consent_given && (
        <Alert 
          severity="info" 
          sx={{ mb: 3 }}
          action={
            <Button 
              color="inherit" 
              size="small"
              onClick={() => setConsentDialogOpen(true)}
            >
              Enable
            </Button>
          }
        >
          <AlertTitle>Communication Clone is Disabled</AlertTitle>
          This feature requires your explicit consent as it analyzes your writing style and can generate text that mimics your voice. 
          Your privacy is important - no data is shared without your permission.
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Style Profile Panel */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <InsightsIcon sx={{ mr: 1 }} /> 
              Your Communication Style
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              This is how the AI perceives your writing style based on analyzed communications.
            </Typography>
            <Divider sx={{ my: 2 }} />
            
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="subtitle2" gutterBottom>
                  Writing Style
                </Typography>
                <Chip 
                  label={styleProfile.summary.writing_style === 'formal' ? 'Formal' : 'Casual'}
                  color={styleProfile.summary.writing_style === 'formal' ? 'primary' : 'secondary'}
                  variant="outlined"
                  sx={{ mb: 2 }}
                />
                
                <Typography variant="subtitle2" gutterBottom>
                  Formality
                </Typography>
                <Slider
                  value={styleProfile.formality_score * 100}
                  valueLabelDisplay="auto"
                  disabled
                  valueLabelFormat={value => `${value}%`}
                />
              </Grid>
              
              <Grid item xs={6}>
                <Typography variant="subtitle2" gutterBottom>
                  Complexity
                </Typography>
                <Chip 
                  label={styleProfile.summary.complexity}
                  color="primary"
                  variant="outlined"
                  sx={{ mb: 2 }}
                />
                
                <Typography variant="subtitle2" gutterBottom>
                  Emoji Usage
                </Typography>
                <Chip 
                  label={styleProfile.summary.emoji_usage}
                  color={styleProfile.summary.emoji_usage === 'frequent' ? 'secondary' : 'default'}
                  variant="outlined"
                />
              </Grid>
              
              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle2" gutterBottom>
                  Your Most Used Phrases
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {styleProfile.summary.sample_phrases.map((phrase, index) => (
                    <Chip 
                      key={index}
                      label={phrase}
                      size="small"
                      variant="outlined"
                    />
                  ))}
                  {styleProfile.summary.sample_phrases.length === 0 && (
                    <Typography variant="body2" color="text.secondary">
                      Not enough data to identify common phrases yet.
                    </Typography>
                  )}
                </Box>
              </Grid>
              
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>
                  Style Analysis Confidence
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Box sx={{ width: '100%', mr: 1 }}>
                    <LinearProgressWithLabel value={styleProfile.summary.mimicry_confidence * 100} />
                  </Box>
                  <Box sx={{ minWidth: 35 }}>
                    <Typography variant="body2" color="text.secondary">{`${Math.round(styleProfile.summary.mimicry_confidence * 100)}%`}</Typography>
                  </Box>
                </Box>
                <Typography variant="caption" color="text.secondary">
                  The more you use the assistant, the more accurate the style analysis becomes.
                </Typography>
              </Grid>
            </Grid>
            
            <Box sx={{ mt: 3 }}>
              <Button
                variant="outlined"
                startIcon={<SampleTextIcon />}
                onClick={() => setGenerationDialogOpen(true)}
                disabled={!settings.enabled || !settings.consent_given}
              >
                Generate Sample Text
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* Settings Panel */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <SettingsIcon sx={{ mr: 1 }} /> 
              Communication Clone Settings
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Configure how the AI learns and mimics your communication style.
            </Typography>
            <Divider sx={{ my: 2 }} />
            
            <FormGroup>
              <FormControlLabel
                control={<Switch 
                  checked={settings.enabled}
                  onChange={handleSettingChange}
                  name="enabled"
                  disabled={!settings.consent_given && !settings.enabled}
                />}
                label="Enable Communication Clone"
              />
              
              <Box sx={{ ml: 3, mt: 2 }}>
                <FormControlLabel
                  control={<Switch 
                    checked={settings.collect_training_data}
                    onChange={handleSettingChange}
                    name="collect_training_data"
                    disabled={!settings.enabled}
                  />}
                  label="Collect Training Data from Your Communications"
                />
                
                <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                  Style Mimicry Level: {Math.round(settings.style_mimicry_level * 100)}%
                </Typography>
                <Slider
                  value={settings.style_mimicry_level}
                  min={0}
                  max={1}
                  step={0.1}
                  disabled={!settings.enabled}
                  valueLabelDisplay="auto"
                  valueLabelFormat={value => `${Math.round(value * 100)}%`}
                  onChange={(_, value) => setSettings({...settings, style_mimicry_level: value})}
                  onChangeCommitted={(_, value) => handleSettingChange({target: {name: 'style_mimicry_level', value}})}
                />
                <Typography variant="caption" color="text.secondary">
                  Higher values produce text that more closely matches your style, lower values allow more creative freedom.
                </Typography>
              </Box>
            </FormGroup>
            
            <Divider sx={{ my: 3 }} />
            
            <Typography variant="subtitle2" gutterBottom>
              Preferred Model
            </Typography>
            <FormControl component="fieldset">
              <RadioGroup
                value={settings.preferred_model}
                onChange={handleModelChange}
                name="preferred_model"
              >
                <FormControlLabel 
                  value="local" 
                  control={<Radio disabled={!settings.enabled} />} 
                  label={
                    <Box>
                      <Typography variant="body2">Local Models</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Runs on your device for better privacy, but less accurate mimicry
                      </Typography>
                    </Box>
                  } 
                />
                <FormControlLabel 
                  value="openai" 
                  control={<Radio disabled={!settings.enabled} />} 
                  label={
                    <Box>
                      <Typography variant="body2">Cloud AI (OpenAI)</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Better quality but requires internet connection
                      </Typography>
                    </Box>
                  } 
                />
                <FormControlLabel 
                  value="rule_based" 
                  control={<Radio disabled={!settings.enabled} />} 
                  label={
                    <Box>
                      <Typography variant="body2">Rule-Based</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Simple pattern matching with no external services or models
                      </Typography>
                    </Box>
                  } 
                />
              </RadioGroup>
            </FormControl>
            
            <Divider sx={{ my: 3 }} />
            
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <SecurityIcon color="action" sx={{ mr: 1 }} />
              <Typography variant="body2" color="text.secondary">
                Your privacy is important. All communication data is stored locally unless you specifically enable cloud services.
              </Typography>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Consent Dialog */}
      <Dialog
        open={consentDialogOpen}
        onClose={() => setConsentDialogOpen(false)}
      >
        <DialogTitle>Enable Communication Clone?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            <Typography paragraph>
              The Communication Clone feature will:
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText 
                  primary="Analyze your writing style" 
                  secondary="Including word choice, sentence structure, and formatting preferences"
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Store samples of your communications" 
                  secondary="Only from contexts you allow (email drafts, chat messages, etc.)"
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Generate text that mimics your style" 
                  secondary="For drafting emails, messages, and other communications"
                />
              </ListItem>
            </List>
            <Typography paragraph sx={{ mt: 2 }}>
              <strong>Your data remains private.</strong> By default, all processing happens locally on your device. 
              You can separately enable cloud AI models if desired.
            </Typography>
            <Typography>
              You can disable this feature at any time and delete all stored communication data from your settings.
            </Typography>
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConsentDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleConsentConfirm} variant="contained">
            I Consent
          </Button>
        </DialogActions>
      </Dialog>

      {/* Text Generation Dialog */}
      <Dialog
        open={generationDialogOpen}
        onClose={() => setGenerationDialogOpen(false)}
        fullWidth
        maxWidth="md"
      >
        <DialogTitle>Generate Text in Your Style</DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ mb: 2 }}>
            Provide a prompt, and the AI will continue it in your communication style.
          </DialogContentText>
          
          <FormControl component="fieldset" sx={{ mb: 2 }}>
            <FormLabel component="legend">Context</FormLabel>
            <RadioGroup
              row
              value={generationContext}
              onChange={(e) => setGenerationContext(e.target.value)}
            >
              <FormControlLabel value="email_draft" control={<Radio />} label="Email" />
              <FormControlLabel value="message_draft" control={<Radio />} label="Message" />
              <FormControlLabel value="note" control={<Radio />} label="Note" />
            </RadioGroup>
          </FormControl>
          
          <TextField
            autoFocus
            label="Your prompt"
            fullWidth
            multiline
            rows={3}
            variant="outlined"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            sx={{ mb: 3 }}
          />
          
          <Button
            variant="contained"
            color="primary"
            onClick={handleGenerateText}
            disabled={generationLoading || !prompt}
            startIcon={generationLoading ? <CircularProgress size={20} /> : <SendIcon />}
            sx={{ mb: 3 }}
          >
            {generationLoading ? 'Generating...' : 'Generate'}
          </Button>
          
          {generatedText && (
            <Paper variant="outlined" sx={{ p: 2, backgroundColor: 'background.default' }}>
              <Typography variant="subtitle2" gutterBottom>
                Generated Text:
              </Typography>
              <Typography variant="body1">
                {generatedText}
              </Typography>
            </Paper>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setGenerationDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

// Helper component for linear progress with label
function LinearProgressWithLabel(props) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
      <Box sx={{ width: '100%', mr: 1 }}>
        <LinearProgress variant="determinate" {...props} />
      </Box>
    </Box>
  );
}

export default CommunicationClone;
