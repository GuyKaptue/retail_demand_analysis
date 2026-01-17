import os
import matplotlib.pyplot as plt # type: ignore
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch # type: ignore
from src.utils import get_path

class Visualizer:
    """Enhanced visualizer with detailed workflow diagrams matching processing logic."""
    
    def __init__(self):
        self.colors = {
            'input': '#ffcccc',      # Light red - raw input
            'process': '#cce5ff',    # Light blue - processing
            'filter': '#d4edda',     # Light green - filtering
            'transform': '#fff3cd',  # Light yellow - transformation
            'output': '#e2d9f3',     # Light purple - output
            'metric': '#ffd9b3',     # Light orange - metrics
        }
    # -------------------------------
    # Distribution plots
    # -------------------------------
    def plot_distribution(self, df, region, top_families=None, items_df=None):
        """
        Plot distribution of stores and families.
        
        Args:
            df: DataFrame with store_nbr and/or item_nbr columns
            region: Region name for title
            top_families: List of top family names (optional)
            items_df: DataFrame with item_nbr and family mapping (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Store distribution (if applicable)
        if 'store_nbr' in df.columns:
            store_counts = df['store_nbr'].value_counts().head(10)
            axes[0].bar(range(len(store_counts)), store_counts.values, 
                       color='#3498db', alpha=0.8, edgecolor='#2c3e50')
            axes[0].set_xlabel('Store Number', fontweight='bold')
            axes[0].set_ylabel('Record Count', fontweight='bold')
            axes[0].set_title(f'Top 10 Stores - {region}', fontweight='bold')
            axes[0].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(store_counts.values):
                axes[0].text(i, v + max(store_counts.values)*0.01, 
                           f'{v:,}', ha='center', va='bottom', fontsize=8)
        else:
            axes[0].text(0.5, 0.5, 'No store data available', 
                        ha='center', va='center', fontsize=12,
                        transform=axes[0].transAxes)
            axes[0].set_title('Store Distribution', fontweight='bold')
        
        # Family distribution (requires merge with items.csv)
        if 'family' in df.columns:
            # Direct family column
            family_counts = df['family'].value_counts().head(10)
        elif 'item_nbr' in df.columns and items_df is not None:
            # Merge with items to get families
            df_merged = df.merge(items_df[['item_nbr', 'family']], 
                               on='item_nbr', how='left')
            family_counts = df_merged['family'].value_counts().head(10)
        else:
            family_counts = None
        
        if family_counts is not None and len(family_counts) > 0:
            bars = axes[1].barh(range(len(family_counts)), family_counts.values, 
                               color='#e74c3c', alpha=0.8, edgecolor='#2c3e50')
            axes[1].set_yticks(range(len(family_counts)))
            axes[1].set_yticklabels(family_counts.index, fontsize=9)
            axes[1].set_xlabel('Record Count', fontweight='bold')
            axes[1].set_title(f'Top 10 Product Families - {region}', fontweight='bold')
            axes[1].grid(axis='x', alpha=0.3)
            
            # Highlight top families if provided
            if top_families:
                for i, family in enumerate(family_counts.index):
                    if family in top_families:
                        bars[i].set_color('#2ecc71')
            
            # Add value labels on bars
            for i, v in enumerate(family_counts.values):
                axes[1].text(v + max(family_counts.values)*0.01, i, 
                           f'{v:,}', ha='left', va='center', fontsize=8)
        else:
            axes[1].text(0.5, 0.5, 'No family data available\n(merge with items.csv needed)', 
                        ha='center', va='center', fontsize=12,
                        transform=axes[1].transAxes)
            axes[1].set_title('Family Distribution', fontweight='bold')
        
        plt.tight_layout()
        
        output_path = os.path.join(
            get_path("loader", week=1), 
            f"distribution_{region}.png"
        )
        #os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        
        print(f"✅ Distribution plot saved to {output_path}")
        plt.show()


    # -------------------------------
    # Daily time series plot
    # -------------------------------
    def plot_daily(self, df_daily, group_by: str = "family"):
        fig, ax = plt.subplots(figsize=(12, 6))
        for key, grp in df_daily.groupby(group_by):
            ax.plot(grp["date"], grp["unit_sales"], label=key)
        ax.set_title(f"Daily Unit Sales by {group_by}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Unit Sales")
        ax.legend()
        plt.tight_layout()
        
        output_path = os.path.join(
            get_path("loader", week=1), 
            f"daily_unit_sales_by_{group_by}.png"
        )
        #os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        
        print(f"✅ Daily unit sales plot saved to {output_path}")   
        plt.show()
        
    def plot_preprocessing_steps(self, df_daily):
        """Visualize preprocessing steps: raw, smoothed, differenced."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        axes[0].plot(df_daily["date"], df_daily["unit_sales"], color="#66b3ff")
        axes[0].set_title("Raw Daily Series")

        df_daily["smoothed"] = df_daily["unit_sales"].rolling(window=7).mean()
        axes[1].plot(df_daily["date"], df_daily["smoothed"], color="#ff9999")
        axes[1].set_title("7-Day Rolling Mean")

        df_daily["diff"] = df_daily["unit_sales"].diff()
        axes[2].plot(df_daily["date"], df_daily["diff"], color="#99ff99")
        axes[2].set_title("Differenced Series")

        plt.tight_layout()
        output_path = os.path.join(
            get_path("loader", week=1), 
            "preprocessing_steps.png"
        )
        #os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Preprocessing steps plot saved to {output_path}")
        plt.show()

    # -------------------------------
    # Workflow as Decision Tree
    # -------------------------------
    def plot_train_subset_workflow(self, region="Pichincha", sample_size=2_000_000, 
                                   chunk_size=1_000_000, top_n=3):
        """
        Visualize the complete workflow matching prepare_train_subset logic.
        Shows: stores.csv + items.csv + train.csv → processing steps → output
        
        Args:
            region: Target region for filtering
            sample_size: Final sample size
            chunk_size: Chunk size for reading train.csv
            top_n: Number of top families to keep
        """
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 12)
        ax.axis("off")

        # ============================================================
        # TITLE
        # ============================================================
        ax.text(7, 11.2, "Train Subset Preparation Pipeline", 
                ha="center", va="center", fontsize=18, fontweight="bold", 
                color="#2c3e50", bbox=dict(boxstyle="round,pad=0.5", 
                facecolor="#ecf0f1", edgecolor="#2c3e50", linewidth=2))

        # ============================================================
        # HELPER FUNCTIONS
        # ============================================================
        def draw_box(x, y, width, height, text, color, style="round,pad=0.3"):
            """Draw a fancy box with text."""
            box = FancyBboxPatch(
                (x, y), width, height,
                boxstyle=style,
                edgecolor="#2c3e50",
                linewidth=2,
                facecolor=color,
                alpha=0.9
            )
            ax.add_patch(box)
            ax.text(x + width/2, y + height/2, text, 
                   ha="center", va="center", fontsize=10, 
                   color="#2c3e50", weight="bold")

        def draw_arrow(x1, y1, x2, y2, label="", curved=False):
            """Draw arrow with optional label."""
            if curved:
                style = "arc3,rad=.3"
            else:
                style = "arc3,rad=0"
            
            arrow = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="->",
                mutation_scale=20,
                color="#34495e",
                linewidth=2,
                connectionstyle=style
            )
            ax.add_patch(arrow)
            
            if label:
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mid_x, mid_y + 0.2, label, 
                       ha="center", va="bottom", fontsize=8, 
                       color="#34495e", style="italic",
                       bbox=dict(boxstyle="round,pad=0.2", 
                                facecolor="white", alpha=0.8))

        def draw_data_icon(x, y, text):
            """Draw data source icon."""
            ax.plot(x, y, 'o', markersize=30, color='#3498db', 
                   markeredgecolor='#2c3e50', markeredgewidth=2)
            ax.text(x, y, text, ha="center", va="center", 
                   fontsize=9, color="white", weight="bold")

        # ============================================================
        # STEP 0: INPUT DATA SOURCES (Top Row)
        # ============================================================
        draw_data_icon(2, 9.5, "CSV")
        draw_box(1, 8.8, 2, 0.6, "stores.csv", self.colors['input'])
        
        draw_data_icon(7, 9.5, "CSV")
        draw_box(6, 8.8, 2, 0.6, "items.csv", self.colors['input'])
        
        draw_data_icon(12, 9.5, "CSV")
        draw_box(10.5, 8.8, 3, 0.6, "train.csv\n(54M+ rows)", self.colors['input'])

        # ============================================================
        # STEP 1: GET REGION STORES (Left Branch)
        # ============================================================
        draw_arrow(2, 8.8, 2, 7.5)
        draw_box(0.5, 6.8, 3, 0.6, 
                f"Get Region Stores\n'{region}'", 
                self.colors['filter'])
        ax.text(2, 6.3, "get_region_stores()", 
               ha="center", fontsize=8, style="italic", color="#7f8c8d")

        # Store IDs output
        draw_arrow(2, 6.8, 2, 6.2)
        draw_box(1, 5.5, 2, 0.6, 
                f"store_ids\n[{region}]", 
                self.colors['metric'])

        # ============================================================
        # STEP 2: GET TOP FAMILIES (Middle Branch)
        # ============================================================
        draw_arrow(7, 8.8, 7, 7.5)
        draw_box(5.5, 6.8, 3, 0.6, 
                f"Get Top {top_n} Families", 
                self.colors['filter'])
        ax.text(7, 6.3, "get_top_families()", 
               ha="center", fontsize=8, style="italic", color="#7f8c8d")
        
        # Top families output
        draw_arrow(7, 6.8, 7, 6.2)
        draw_box(6, 5.5, 2, 0.6, 
                f"top_{top_n}_families", 
                self.colors['metric'])

        # ============================================================
        # STEP 3: CHUNKED READING (Right Branch)
        # ============================================================
        draw_arrow(12, 8.8, 12, 7.5)
        draw_box(10.5, 6.8, 3, 0.6, 
                f"Chunked Reading\n({chunk_size:,} rows)", 
                self.colors['process'])
        ax.text(12, 6.3, "filter_train_chunks()", 
               ha="center", fontsize=8, style="italic", color="#7f8c8d")

        # ============================================================
        # STEP 4: FILTER BY STORES (Converge Step 1 + 3)
        # ============================================================
        draw_arrow(2.5, 5.2, 9.5, 4.5, "store_ids")
        draw_arrow(12, 6.8, 11, 4.5)
        
        draw_box(9.5, 3.8, 3, 0.6, 
                "Filter by Store IDs", 
                self.colors['filter'])
        ax.text(11, 3.3, "chunk[store_nbr.isin()]", 
               ha="center", fontsize=8, style="italic", color="#7f8c8d")

        # ============================================================
        # STEP 5: FILTER TOP FAMILIES (Converge Step 2 + 4)
        # ============================================================
        draw_arrow(8, 5.2, 8, 3.5, "families", curved=True)
        draw_arrow(11, 3.8, 8.5, 2.9)
        
        draw_box(7, 2.2, 3, 0.6, 
                f"Filter Top {top_n} Families", 
                self.colors['filter'])
        ax.text(8.5, 1.7, "filter_top_families()", 
               ha="center", fontsize=8, style="italic", color="#7f8c8d")

        # ============================================================
        # STEP 6: SAMPLING
        # ============================================================
        draw_arrow(8.5, 2.2, 8.5, 1.5)
        draw_box(7, 0.8, 3, 0.6, 
                f"Sample {sample_size:,} Rows", 
                self.colors['transform'])
        ax.text(8.5, 0.3, "sample_subset()", 
               ha="center", fontsize=8, style="italic", color="#7f8c8d")

        # ============================================================
        # STEP 7: SAVE OUTPUT
        # ============================================================
        draw_arrow(10, 1.1, 12, 1.1)
        draw_box(12, 0.8, 2, 0.6, 
                "Save CSV", 
                self.colors['output'])
        ax.text(13, 0.3, "train_subset.csv", 
               ha="center", fontsize=8, style="italic", color="#7f8c8d")

        # ============================================================
        # ANNOTATIONS & METRICS
        # ============================================================
        # Add processing stats box
        stats_text = (
            f"Pipeline Configuration:\n"
            f"• Region: {region}\n"
            f"• Top Families: {top_n}\n"
            f"• Chunk Size: {chunk_size:,}\n"
            f"• Sample Size: {sample_size:,}\n"
            f"• Output: train_subset.csv"
        )
        ax.text(0.5, 2.5, stats_text, 
               ha="left", va="top", fontsize=9,
               bbox=dict(boxstyle="round,pad=0.5", 
                        facecolor="#ecf0f1", 
                        edgecolor="#2c3e50",
                        linewidth=1.5),
               family="monospace")

        # Add legend
        legend_x, legend_y = 0.5, 4.5
        legend_items = [
            ("Input Data", self.colors['input']),
            ("Processing", self.colors['process']),
            ("Filtering", self.colors['filter']),
            ("Transform", self.colors['transform']),
            ("Output", self.colors['output']),
        ]
        
        ax.text(legend_x, legend_y + 0.5, "Legend:", 
               fontsize=10, weight="bold", color="#2c3e50")
        
        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y - (i * 0.3)
            draw_box(legend_x, y_pos - 0.15, 0.4, 0.25, "", color)
            ax.text(legend_x + 0.6, y_pos, label, 
                   ha="left", va="center", fontsize=8, color="#2c3e50")

        # ============================================================
        # SAVE AND DISPLAY
        # ============================================================
        plt.tight_layout()
        
        output_path = os.path.join(
            get_path("loader", week=1), 
            "train_subset_workflow_detailed.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight", 
                   facecolor="white", edgecolor="none")
        
        print(f"✅ Detailed workflow diagram saved to {output_path}")
        plt.show()

    