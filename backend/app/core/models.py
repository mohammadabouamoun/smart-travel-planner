from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from backend.app.core.database import Base, Vector
import uuid

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AgentRun(Base):
    __tablename__ = "agent_runs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    query = Column(Text, nullable=False)
    answer = Column(Text)
    tools_fired = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class DestinationDocument(Base):
    __tablename__ = "destination_documents"
    id = Column(Integer, primary_key=True)
    destination_name = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384))