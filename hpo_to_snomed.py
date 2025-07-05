"""
hpo_to_snomed.py

Converts disease-phenotype associations into disease-SNOMED associations using:
- official HPO (terminology,disease, gene association)
    download link: https://github.com/obophenotype/human-phenotype-ontology/releases
    phenotype.hpoa : https://github.com/obophenotype/human-phenotype-ontology/blob/fd050e5aea1d01bd3e6b4e524acb7289e1f3e266/docs/annotations/genes_to_phenotype.md
    phenotype_to_genes.txt : https://github.com/obophenotype/human-phenotype-ontology/blob/fd050e5aea1d01bd3e6b4e524acb7289e1f3e266/docs/annotations/phenotype_to_genes.md
    genes_to_phenotype.txt : https://github.com/obophenotype/human-phenotype-ontology/blob/fd050e5aea1d01bd3e6b4e524acb7289e1f3e266/docs/annotations/genes_to_phenotype.md
- official SNOMED release (2025.01.01)
- SapBERT model
"""
import os
import pickle
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import obonet
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ==========================
# CONFIGURATION
# ==========================
@dataclass
class Config:
    """Configuration class for file paths and model settings."""
    hpo_files_path: str = "docs/"
    snomed_file_path: str = "snomed/"
    embedding_model_hf: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    output_embedding_folder: str = "embeddings_sapbert_and_info/"
    output_mapping_folder: str = "output_mapping/"
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mapping_threshold: float = 0.8
    batch_size: int = 1000
    top_k: int = 1

class DataLoader:
    """Handles loading and preprocessing of HPO and SNOMED data."""
    def __init__(self, config: Config):
        self.config =  config

    def read_txt(self,file_name) -> pl.DataFrame:
        return pl.read_csv(file_name, separator="\t", comment_prefix = "#")
    
    def read_parquet(self, file_name, cols) -> pl.DataFrame:
        return pl.read_parquet(file_name, columns=cols)
    
    def load_hpo_data(self):
        # Load ontology
        graph = obonet.read_obo(f"{self.config.hpo_files_path}hp.obo")
    
        # Load phenotype-disease mappings
        df_hpoa = self.read_txt(f"{self.config.hpo_files_path}phenotype.hpoa")
        df_p2g = self.read_txt(f"{self.config.hpo_files_path}phenotype_to_genes.txt")
        df_g2p = self.read_txt(f"{self.config.hpo_files_path}genes_to_phenotype.txt")
        
        return graph, df_hpoa, df_p2g, df_g2p
    
    def load_snomed_data(self):
        # get release data (for label and synonym)
        df_snomed_release = self.read_parquet(f"{self.config.snomed_file_path}released_version.parquet", ['id',"term"])

        # get our snomed data (for semantic tag / top category), and limit to only findings
        df_concept_snomed = self.read_parquet(f"{self.config.snomed_file_path}concept_snomed_hug.parquet", ['id',"label","concept_type","top_category"])
        df_concept_snomed = df_concept_snomed.filter((pl.col("concept_type") =="SCT_PRE") & (pl.col("top_category") == "finding"))
        finding_snomed = df_concept_snomed['id'].to_list()
        return df_snomed_release, finding_snomed

class HPOProcessor:
    """Processes HPO ontology data and extracts concept information."""
    
    def __init__(self, config: Config):
        self.config = config

    def extract_hpo_concepts(self, graph: object) -> pl.DataFrame:
        """Extract HPO concepts from ontology graph."""
        print("Extracting HPO concepts...")
        
        node_rows = []
        for node_id, data in graph.nodes(data=True):
            node_rows.append({
                "hpo_id": node_id,
                "label": data.get("name", None),
                "def": data.get("def", None),
                "comment": data.get("comment", None),
                "synonyms": data.get("synonym", []),
                "xrefs": data.get("xref", []),
                "is_a": data.get("is_a", []),
                "is_obsolete": data.get("is_obsolete", "false")
            })
        
        return pl.DataFrame(node_rows).filter(pl.col("is_obsolete") == "false")
    
    def separate_mappable_concepts(self, df_concept_hpo: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Separate HPO concepts into mappable and non-mappable based on SNOMED xrefs."""
        print("Separating mappable and non-mappable HPO concepts...")
        
        # Get HPO concepts that have SNOMED cross-references
        concept_hpo_mappable = df_concept_hpo.explode("xrefs").filter(
            pl.col("xrefs").str.contains("SNOMED")
        )["hpo_id"].unique()
        
        # Split into mappable and non-mappable
        df_mappable = df_concept_hpo.filter(pl.col("hpo_id").is_in(concept_hpo_mappable))
        df_not_mappable = df_concept_hpo.filter(~pl.col("hpo_id").is_in(concept_hpo_mappable))
        
        return df_mappable, df_not_mappable
    
class OfficialMappingProcessor:
    """Processes official HPO to SNOMED mappings."""
    
    def create_official_mappings(self, df_mappable: pl.DataFrame, df_snomed_release: pl.DataFrame) -> pl.DataFrame:
        """Create official mappings from HPO cross-references to SNOMED."""
        print("Creating official mappings...")
        
        # Process SNOMED cross-references
        df_processed = (df_mappable
                       .select("hpo_id", "label", "xrefs")
                       .explode("xrefs")
                       .filter(pl.col("xrefs").str.contains("SNOMEDCT"))
                       .with_columns(pl.col("xrefs").str.split(":"))
                       .explode("xrefs")
                       .filter(~pl.col("xrefs").str.contains("SNOMEDCT")))
        
        # Join with SNOMED release data
        df_official_mappings = df_processed.join(
            df_snomed_release, 
            left_on="xrefs", 
            right_on="id"
        ).rename({
            "label": "hpo_label", 
            "xrefs": "snomed_id", 
            "term": "snomed_label"
        })
        
        return df_official_mappings
    
class EmbeddingProcessor:
    """Handles embedding generation and similarity computation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
    
    def load_model(self) -> SentenceTransformer:
        """Load the SentenceTransformer model."""
        print(f"Loading embedding model: {self.config.embedding_model_hf}")
        self.model = SentenceTransformer(self.config.embedding_model_hf, device=self.config.device)
        return self.model
    
    def generate_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if self.model is None:
            self.load_model()
        
        return self.model.encode(texts, show_progress_bar=show_progress)
    
    def top_k_similarity_batch(self, query_matrix: np.ndarray, candidate_matrix: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute top-k similarity scores between query and candidate matrices in batches."""
        print("Computing similarity scores...")
        
        # Convert to torch tensors
        if not isinstance(query_matrix, torch.Tensor):
            query_matrix = torch.tensor(query_matrix, dtype=torch.float32, device=self.config.device)
        if not isinstance(candidate_matrix, torch.Tensor):
            candidate_matrix = torch.tensor(candidate_matrix, dtype=torch.float32, device=self.config.device)
        
        # Normalize vectors
        query_matrix = nn.functional.normalize(query_matrix, p=2, dim=1)
        candidate_matrix = nn.functional.normalize(candidate_matrix, p=2, dim=1)
        candidate_matrix_T = candidate_matrix.T
        
        num_concepts = query_matrix.shape[0]
        num_batches = (num_concepts + self.config.batch_size - 1) // self.config.batch_size
        
        all_top_scores = []
        all_top_indices = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, num_concepts)
            
            print(f"Processing batch {batch_idx + 1}/{num_batches} ({start_idx}-{end_idx})")
            
            query_batch = query_matrix[start_idx:end_idx]
            scores = torch.matmul(query_batch, candidate_matrix_T)
            top_scores, top_indices = torch.topk(scores, k=self.config.top_k, dim=1)
            
            all_top_scores.append(top_scores.cpu())
            all_top_indices.append(top_indices.cpu())
            
            # Free memory
            del scores, top_scores, top_indices
            torch.cuda.empty_cache()
        
        return torch.cat(all_top_scores, dim=0), torch.cat(all_top_indices, dim=0)

class SemanticMappingProcessor:
    """Creates semantic mappings using embeddings."""
    
    def __init__(self, config: Config):
        self.config = config
        self.embedding_processor = EmbeddingProcessor(config)
    
    def create_semantic_mappings(self, df_source: pl.DataFrame, df_target: pl.DataFrame, 
                                source_id_col: str, source_label_col: str,
                                target_id_col: str, target_label_col: str) -> pl.DataFrame:
        """Create semantic mappings between source and target concepts."""
        print(f"Creating semantic mappings from {source_id_col} to {target_id_col}...")
        
        # Prepare source data
        df_source_indexed = df_source.clone()
        df_source_indexed.insert_column(0, pl.Series(np.arange(len(df_source_indexed))).alias("index_source"))
        source_labels = df_source_indexed[source_label_col].to_list()
        index2source = dict(zip(df_source_indexed["index_source"], df_source_indexed[source_id_col]))
        
        # Prepare target data
        df_target_indexed = df_target.clone()
        df_target_indexed.insert_column(0, pl.Series(np.arange(len(df_target_indexed))).alias("index_target"))
        target_labels = df_target_indexed[target_label_col].to_list()
        index2target = dict(zip(df_target_indexed["index_target"], df_target_indexed[target_id_col]))
        
        # Generate embeddings
        source_embeddings = self.embedding_processor.generate_embeddings(source_labels)
        target_embeddings = self.embedding_processor.generate_embeddings(target_labels)
        
        # Compute similarities
        top_scores, top_indices = self.embedding_processor.top_k_similarity_batch(
            source_embeddings, target_embeddings
        )
        
        # Create mapping dataframe
        mappings = []
        for source_idx in range(len(top_scores)):
            score = top_scores[source_idx, 0].item()
            target_idx = top_indices[source_idx, 0].item()
            
            source_id = index2source[source_idx]
            target_id = index2target[target_idx]
            
            source_label = df_source_indexed.filter(pl.col(source_id_col) == source_id)[source_label_col].to_list()
            target_label = df_target_indexed.filter(pl.col(target_id_col) == target_id)[target_label_col].to_list()
            
            mappings.append({
                f"{source_id_col}": source_id,
                f"{source_label_col}": source_label,
                f"{target_id_col}": target_id,
                f"{target_label_col}": target_label,
                "score_top_1": score,
                "mapping_type": "sapbert"
            })
        
        return pl.DataFrame(mappings)
    
class FileManager:
    """Manages file I/O operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.config.output_embedding_folder, exist_ok=True)
        os.makedirs(self.config.output_mapping_folder, exist_ok=True)
    
    def save_embeddings(self, embeddings: np.ndarray, filename: str):
        """Save embeddings to file."""
        filepath = os.path.join(self.config.output_embedding_folder, filename)
        np.save(filepath, embeddings)
        print(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filename: str) -> np.ndarray:
        """Load embeddings from file."""
        filepath = os.path.join(self.config.output_embedding_folder, filename)
        return np.load(filepath)
    
    def save_dataframe(self, df: pl.DataFrame, filename: str, folder: str = None):
        """Save dataframe to parquet file."""
        folder = folder or self.config.output_mapping_folder
        filepath = os.path.join(folder, filename)
        df.write_parquet(filepath)
        print(f"Saved dataframe to {filepath}")
    
    def load_dataframe(self, filename: str, folder: str = None) -> pl.DataFrame:
        """Load dataframe from parquet file."""
        folder = folder or self.config.output_mapping_folder
        filepath = os.path.join(folder, filename)
        return pl.read_parquet(filepath)
    
    def save_pickle(self, obj: object, filename: str):
        """Save object to pickle file."""
        filepath = os.path.join(self.config.output_embedding_folder, filename)
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
        print(f"Saved object to {filepath}")
    
    def load_pickle(self, filename: str) -> object:
        """Load object from pickle file."""
        filepath = os.path.join(self.config.output_embedding_folder, filename)
        with open(filepath, "rb") as f:
            return pickle.load(f)


class HPOSNOMEDMappingPipeline:
    """Main pipeline for HPO to SNOMED mapping."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.hpo_processor = HPOProcessor(config)
        self.official_mapper = OfficialMappingProcessor()
        self.semantic_mapper = SemanticMappingProcessor(config)
        self.file_manager = FileManager(config)
    
    def run_full_pipeline(self):
        """Execute the complete mapping pipeline."""
        print("Starting HPO to SNOMED mapping pipeline...")
        
        # Load data
        graph, df_hpoa, df_p2g, df_g2p = self.data_loader.load_hpo_data()
        df_snomed_release, finding_snomed = self.data_loader.load_snomed_data()
        
        # Process HPO concepts
        df_concept_hpo = self.hpo_processor.extract_hpo_concepts(graph)
        df_mappable, df_not_mappable = self.hpo_processor.separate_mappable_concepts(df_concept_hpo)
        
        # Create official mappings
        df_official_mappings = self.official_mapper.create_official_mappings(df_mappable, df_snomed_release)
        
        # Filter SNOMED for findings
        df_snomed_findings = df_snomed_release.filter(pl.col("id").is_in(finding_snomed))
        
        # Create semantic mappings for non-mappable HPO concepts
        df_semantic_hpo_mappings = self.semantic_mapper.create_semantic_mappings(
            df_not_mappable, df_snomed_findings,
            "hpo_id", "label", "id", "term"
        )
        
        df_semantic_hpo_mappings = df_semantic_hpo_mappings.rename({"label": "hpo_label", "id": "snomed_id", "term" : "snomed_label"})

        # Combine official and semantic mappings
        df_official_formatted = df_official_mappings.with_columns([
            pl.lit(1.0).alias("score_top_1"),
            pl.col("hpo_label").map_elements(lambda x: [x], return_dtype=pl.List(pl.String)),
            pl.col("snomed_label").map_elements(lambda x: [x], return_dtype=pl.List(pl.String)),
            pl.lit("official").alias("mapping_type")
        ])
        
        df_all_hpo_mappings = pl.concat([df_semantic_hpo_mappings, df_official_formatted], how="vertical")
        
        # Create disease mappings
        df_diseases = df_hpoa.select("database_id", "disease_name").unique()
        df_disease_mappings = self.semantic_mapper.create_semantic_mappings(
            df_diseases, df_snomed_findings,
            "database_id", "disease_name", "id", "term"
        )
        df_disease_mappings = df_disease_mappings.rename({"id": "snomed_id", "term" : "snomed_label"})
        # Save mappings
        self.file_manager.save_dataframe(df_all_hpo_mappings, "all_pheno_hpo_snomed_mapping.parquet")
        self.file_manager.save_dataframe(df_disease_mappings, "all_disease_hpo_snomed_mapping.parquet")
        
        # Create association mappings
        self._create_association_mappings(df_p2g, df_g2p, df_all_hpo_mappings, df_disease_mappings)
        
        print("Pipeline completed successfully!")
    
    def _create_association_mappings(self, df_p2g: pl.DataFrame, df_g2p: pl.DataFrame, 
                                   pheno_mapping: pl.DataFrame, disease_mapping: pl.DataFrame):
        """Create gene-phenotype-disease association mappings."""
        print("Creating association mappings...")
        
        # Filter mappings by threshold
        pheno_filtered = pheno_mapping.filter(pl.col("score_top_1") >= self.config.mapping_threshold)
        disease_filtered = disease_mapping.filter(pl.col("score_top_1") >= self.config.mapping_threshold)
        disease_filtered = disease_filtered.rename({"database_id" :  "disease_id"})
        # Create phenotype to gene mappings
        pheno_to_gene = (df_p2g
                        .join(pheno_filtered, on="hpo_id")
                        .select("snomed_id", "hpo_name", "gene_symbol", "disease_id")
                        .rename({"snomed_id": "pheno_snomed_id"})
                        .join(disease_filtered, on="disease_id")
                        .select("pheno_snomed_id", "hpo_name", "gene_symbol", "snomed_id", "snomed_label")
                        .rename({"snomed_id": "disease_snomed_id", "snomed_label": "disease_snomed_label"})
                        .unique())
        
        # Create gene to phenotype mappings
        gene_to_pheno = (df_g2p
                        .join(pheno_filtered, on="hpo_id")
                        .select("snomed_id", "hpo_name", "gene_symbol", "disease_id")
                        .rename({"snomed_id": "pheno_snomed_id"})
                        .join(disease_filtered, on="disease_id")
                        .select("pheno_snomed_id", "hpo_name", "gene_symbol", "snomed_id", "snomed_label")
                        .rename({"snomed_id": "disease_snomed_id", "snomed_label": "disease_snomed_label"})
                        .unique())
        
        # Save association mappings
        self.file_manager.save_dataframe(pheno_to_gene, "pheno_to_gene_snomed_mapping.parquet")
        self.file_manager.save_dataframe(gene_to_pheno, "gene_to_pheno_snomed_mapping.parquet")


def main():
    """Main function to run the pipeline."""
    # Configuration
    config = Config(
        hpo_files_path="docs/",
        snomed_file_path="snomed/",
        output_embedding_folder="embeddings_sapbert_and_info/",
        output_mapping_folder="output_mapping/",
        mapping_threshold=0.8,
        batch_size=1000
    )
    
    # Run pipeline
    pipeline = HPOSNOMEDMappingPipeline(config)
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()