import { createContext } from 'react';

// Create a theme context with default values
const ThemeContext = createContext({
  mode: 'light',
  toggleTheme: () => {},
});

export default ThemeContext;
