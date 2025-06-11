# main.py
import sys
import logging
from datetime import datetime

from rag_system.pipeline import RAGPipeline
from rag_system.config import config

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_interactive_mode(rag: RAGPipeline):
    """Starts the interactive question-and-answer loop."""
    print("\n💬 Ask me anything about your documents (type 'exit' to quit)")
    print("-" * 60)
    while True:
        try:
            question = input("🤔 Your question: ").strip()
            if not question:
                continue
            if question.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break

            print("\n🔍 Searching and generating answer...")
            start_time = datetime.now()
            result = rag.query(question)
            end_time = datetime.now()

            print("\n💡 Answer:")
            print("-" * 40)
            print(result['answer'])
            
            if result['sources']:
                print(f"\n📖 Sources ({len(result['sources'])} found):")
                print("-" * 40)
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source['file']} (Page {source['page']}) - Score: {source['score']}")
            
            print(f"\n⚡ Response generated in {(end_time - start_time).total_seconds():.2f} seconds.")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in interactive mode: {e}", exc_info=True)
            print(f"An unexpected error occurred: {e}")


def main():
    """Main function to initialize and run the RAG system."""
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'rebuild':
        force_rebuild = True
        print("🔧 Force-rebuilding the index...")
    else:
        force_rebuild = False

    logger.info("🚀 Optimized RAG System Starting...")
    rag_pipeline = RAGPipeline()

    if not rag_pipeline.initialize(force_rebuild=force_rebuild):
        logger.critical("Failed to initialize the RAG pipeline. Exiting.")
        print("❌ Failed to initialize the RAG system. Please check logs for details.")
        return

    stats = rag_pipeline.get_stats()
    print("="*60)
    print("✅ System Initialized Successfully!")
    print(f"📚 Loaded {stats['total_chunks']} chunks from {len(stats['pdf_files'])} PDF(s).")
    print(f"🧠 Embedding Model: {stats['embedding_model']}")
    print(f"🔍 Search Mode: {stats['search_mode']}")
    print("="*60)
    
    run_interactive_mode(rag_pipeline)


if __name__ == "__main__":
    main()