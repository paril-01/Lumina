import React from 'react';
import { Box } from '@mui/material';
import SmartToyIcon from '@mui/icons-material/SmartToy';

function Logo({ size = 40 }) {
  return (
    <Box sx={{ 
      width: size, 
      height: size, 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center',
      backgroundColor: 'primary.main',
      borderRadius: '50%',
      color: 'white'
    }}>
      <SmartToyIcon sx={{ fontSize: size * 0.6 }} />
    </Box>
  );
}

export default Logo;
