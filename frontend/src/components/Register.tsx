import { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate, Link } from 'react-router-dom';

const Register = () => {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    try {
      await register(email, password);
      navigate('/login');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Registration failed');
    }
  };

  return (
    <div style={{ maxWidth: 400, margin: '40px auto' }}>
      <h2>Register</h2>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      <form onSubmit={handleSubmit}>
        <div>
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            required
            style={{ width: '100%', padding: 8, marginBottom: 10 }}
          />
        </div>
        <div>
          <input
            type="password"
            placeholder="Password (min 8 chars)"
            value={password}
            onChange={e => setPassword(e.target.value)}
            required
            minLength={8}
            style={{ width: '100%', padding: 8, marginBottom: 10 }}
          />
        </div>
        <button type="submit" style={{ width: '100%', padding: 10 }}>
          Register
        </button>
      </form>
      <p style={{ marginTop: 15 }}>
        Already have an account? <Link to="/login">Login</Link>
      </p>
    </div>
  );
};

export default Register;