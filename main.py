from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.chat import router as chat_router
from routes.training import router as training_router
from routes.faq import router as faq_router

# Initialize FastAPI app
app = FastAPI(
    title="Customer Support Resolution API",
    description="API for managing customer support tickets and resolutions with RAG knowledge base",
    version="2.0.0"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Register routes
app.include_router(chat_router, tags=["chat"])
app.include_router(training_router, tags=["training"])
app.include_router(faq_router, tags=["faq"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Customer Support Resolution API",
        "version": "2.0.0",
        "endpoints": {
            "chat": "/chat",
            "training": "/training",
            "faq": "/faq",
            "health": "/chat/health",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

