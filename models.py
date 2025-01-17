from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import bcrypt
from datetime import datetime
import os

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    deepseek_api_key = Column(String)
    groq_api_key = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    
    def set_password(self, password):
        salt = bcrypt.gensalt()
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    messages = Column(JSON)  # Store chat messages as JSON
    suggestions = Column(JSON)  # Store current suggestions as JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="conversations")

def init_db():
    """Initialize the database and create tables."""
    # Create database directory if it doesn't exist
    db_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    # Create database engine with absolute path
    db_path = os.path.join(db_dir, 'historical_chat.db')
    engine = create_engine(f'sqlite:///{db_path}')
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    return engine

# Initialize database and create session factory
engine = init_db()
Session = sessionmaker(bind=engine)

def get_db_session():
    """Get a new database session."""
    try:
        session = Session()
        return session
    except Exception as e:
        print(f"Error creating database session: {str(e)}")
        raise 