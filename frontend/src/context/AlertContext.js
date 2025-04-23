import { createContext } from 'react';

const AlertContext = createContext({
  showAlert: () => {}
});

export default AlertContext;
