import React from 'react';
import { Box, Paper, Typography, Divider, Container, Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';

function PrivacyPolicy() {
  const navigate = useNavigate();

  return (
    <Container maxWidth="md">
      <Paper elevation={2} sx={{ p: 4, my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Privacy Policy
        </Typography>
        
        <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
          Effective Date: April 22, 2025
        </Typography>
        
        <Divider sx={{ mb: 3 }} />
        
        <Typography variant="body1" paragraph>
          Your privacy is of utmost importance to us. This Privacy Policy explains how your personal information 
          is collected, used, and disclosed by the Personal AI Assistant.
        </Typography>
        
        <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
          1. Information We Collect
        </Typography>
        <Typography variant="body1" paragraph>
          Our AI Assistant collects data about your activities on your device to learn your behavior patterns and automate tasks. This includes:
        </Typography>
        <ul>
          <li>
            <Typography variant="body1">
              <strong>Application usage:</strong> Which applications you use and when you use them
            </Typography>
          </li>
          <li>
            <Typography variant="body1">
              <strong>Mouse and keyboard interactions:</strong> How you interact with applications (without recording the actual content of what you type)
            </Typography>
          </li>
          <li>
            <Typography variant="body1">
              <strong>Time patterns:</strong> When you perform certain activities
            </Typography>
          </li>
          <li>
            <Typography variant="body1">
              <strong>System information:</strong> Basic information about your operating system and hardware
            </Typography>
          </li>
        </ul>
        
        <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
          2. How We Use Your Information
        </Typography>
        <Typography variant="body1" paragraph>
          We use the collected information to:
        </Typography>
        <ul>
          <li>
            <Typography variant="body1">
              Learn your behavior patterns and preferences
            </Typography>
          </li>
          <li>
            <Typography variant="body1">
              Create personalized automation suggestions
            </Typography>
          </li>
          <li>
            <Typography variant="body1">
              Execute automated tasks on your behalf when enabled
            </Typography>
          </li>
          <li>
            <Typography variant="body1">
              Improve the assistant's functionality and accuracy
            </Typography>
          </li>
        </ul>
        
        <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
          3. Data Storage and Retention
        </Typography>
        <Typography variant="body1" paragraph>
          <strong>All data is stored locally on your device.</strong> We do not transmit your personal data to external servers.
        </Typography>
        <Typography variant="body1" paragraph>
          Raw activity data is automatically deleted after 24 hours once it has been processed by the learning model.
          The behavioral models (learned patterns) are retained until you choose to delete them or uninstall the application.
        </Typography>
        
        <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
          4. Privacy Controls
        </Typography>
        <Typography variant="body1" paragraph>
          You have complete control over your privacy with the following options:
        </Typography>
        <ul>
          <li>
            <Typography variant="body1">
              <strong>Monitoring settings:</strong> Enable or disable activity monitoring
            </Typography>
          </li>
          <li>
            <Typography variant="body1">
              <strong>Learning settings:</strong> Control whether the assistant learns from your activities
            </Typography>
          </li>
          <li>
            <Typography variant="body1">
              <strong>Application exclusions:</strong> Specify applications that should never be monitored
            </Typography>
          </li>
          <li>
            <Typography variant="body1">
              <strong>Data retention:</strong> Adjust how long raw data is kept before being deleted
            </Typography>
          </li>
          <li>
            <Typography variant="body1">
              <strong>Data deletion:</strong> Delete all collected data and learned patterns at any time
            </Typography>
          </li>
        </ul>
        
        <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
          5. Sensitive Information
        </Typography>
        <Typography variant="body1" paragraph>
          The assistant is designed to automatically filter and exclude the collection of:
        </Typography>
        <ul>
          <li>
            <Typography variant="body1">
              Passwords and authentication credentials
            </Typography>
          </li>
          <li>
            <Typography variant="body1">
              Content from banking and financial applications
            </Typography>
          </li>
          <li>
            <Typography variant="body1">
              Personal identifiable information (PII)
            </Typography>
          </li>
          <li>
            <Typography variant="body1">
              Content from applications marked as sensitive in settings
            </Typography>
          </li>
        </ul>
        
        <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
          6. Changes to This Policy
        </Typography>
        <Typography variant="body1" paragraph>
          We may update this Privacy Policy from time to time. We will notify you of any changes by posting the new 
          Privacy Policy on this page and updating the effective date.
        </Typography>
        
        <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
          7. Contact Us
        </Typography>
        <Typography variant="body1" paragraph>
          If you have any questions about this Privacy Policy, please contact us at privacy@ai-assistant.app
        </Typography>
        
        <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
          <Button variant="contained" color="primary" onClick={() => navigate(-1)}>
            Back
          </Button>
        </Box>
      </Paper>
    </Container>
  );
}

export default PrivacyPolicy;
