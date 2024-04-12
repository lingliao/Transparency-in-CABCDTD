###################################################################
# Plot the full image size and cropped area per 598 by 598 pixels #
###################################################################

# Install the package and load the library
install.packages("tidyverse")
library(tidyverse)

########################
#1. full image size

# Read the CSV file into a data frame
df <- read.csv("/content/heaght_width_FULL.csv")

df_processed <- df %>%
  mutate(pathology_processed = case_when(
    pathology == "MALIGNANT" ~ "Malignant",
    pathology == "BENIGN_WITHOUT_CALLBACK" ~ "Benign",
    pathology == "BENIGN" ~ "Benign",
    TRUE ~ as.character(pathology)  # 如果没有匹配到上述条件，保持不变
  ))

# Calculate the total number of dots
total_dots <- nrow(df_processed)
# Format the total_dots with commas every 1000
formatted_total_dots <- format(total_dots, big.mark = ",")

# Calculate the counts for benign and malignant
benign_count <- sum(df_processed$pathology_processed == "Benign")
malignant_count <- sum(df_processed$pathology_processed == "Malignant")

# Create the dot plot with custom gridlines, axis lines, and truncated axes
full_pixel <- ggplot(df_processed, aes(x = Width, y = Height, color = pathology_processed)) +
  geom_point() +
  labs(x = "Width Pixel", y = "Height Pixel", color = "Pathology") +
  scale_color_manual(values = c("#eb6a4d", "#6da9ed")) +
  scale_linetype_manual(values = c("solid", "dashed")) +
  scale_x_continuous(breaks = seq(2000, max(df_processed$Width), by = 1000)) +
  scale_y_continuous(breaks = seq(1500, max(df_processed$Height), by = 1000)) +
  geom_hline(yintercept = 0, color = "grey") +  # Horizontal axis line
  geom_vline(xintercept = 0, color = "grey") +  # Vertical axis line
  geom_hline(yintercept = seq(1500, max(df_processed$Height), by = 1000), color = "grey", linetype = "dashed") +  # Horizontal gridlines
  geom_vline(xintercept = seq(2000, max(df_processed$Width), by = 1000), color = "grey", linetype = "dashed") +  # Vertical gridlines
  coord_cartesian(xlim = c(1500, max(df_processed$Width)), ylim = c(3500, max(df_processed$Height))) +  # Truncate axes
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5)
  ) +
  ggtitle(paste("Original Pixel Information for", formatted_total_dots, "CBIS-DDSM Mass Full Images \nBenign:", benign_count, ", Malignant:", malignant_count))

# Save the plot as a PDF with specified dimensions and resolution
ggsave("full_pixel.pdf", full_pixel, width = 7, height = 5.7, dpi = 1000)

########################
#2. cropped area percentage

# Read the CSV file into a data frame
roi <- read.csv("/content/content/metadata/598_percentage_all.csv")

roi_processed <- roi %>%
  mutate(pathology_processed = case_when(
    pathology == "MALIGNANT" ~ "Malignant",
    pathology == "BENIGN_WITHOUT_CALLBACK" ~ "Benign",
    pathology == "BENIGN" ~ "Benign",
    TRUE ~ as.character(pathology)  # 如果没有匹配到上述条件，保持不变
  ))


# Perform Mann-Whitney U test between benign and malignant groups for area_percentage
wilcox_test_result <- wilcox.test(area_percentage ~ pathology_processed, data = roi_processed)

# Get the p-value from the Mann-Whitney U test result
p_value_wilcox <- wilcox_test_result$p.value

# Print the p-value
print(p_value_wilcox)


# Calculate summary statistics
summary_stats <- roi_processed %>%
  group_by(pathology_processed) %>%
  summarise(
    count = n(),  # Total count
    median = median(area_percentage),
    # mean = mean(area_percentage),  # Calculate mean
    q1 = quantile(area_percentage, 0.25),
    q3 = quantile(area_percentage, 0.75)
  )

# Calculate summary statistics
summary_stats <- roi_processed %>%
  group_by(pathology_processed) %>%
  summarise(
    count = n(),  # Total count
    median = median(area_percentage),
    q1 = quantile(area_percentage, 0.25),
    q3 = quantile(area_percentage, 0.75)
  )

# Calculate the total number of dots
total_dots_ROI <- nrow(roi_processed)
# Format the total_dots with commas every 1000
formatted_total_dots_ROI <- format(total_dots_ROI, big.mark = ",")


# Get counts for malignant and benign
malignant_count <- summary_stats$count[summary_stats$pathology_processed == "Malignant"]
benign_count <- summary_stats$count[summary_stats$pathology_processed == "Benign"]

# Create the violin plot with summary statistics and annotations
ROI_area <- ggplot(roi_processed, aes(x = pathology_processed, y = area_percentage, color = pathology_processed)) +
  geom_violin(trim = FALSE) +  # Create violin plot without trimming
  stat_summary(fun.data = function(x) {
    median <- median(x)
    q1 <- quantile(x, 0.25)
    q3 <- quantile(x, 0.75)
    data.frame(y = c(median, q1, q3), label = c("Median", "1st Quartile", "3rd Quartile"))
  }, geom = "point", shape = 21, size = 3, color = "black") +  # Add summary statistics as points
  geom_hline(data = summary_stats, aes(yintercept = median, color = pathology_processed), linetype = "dashed", size = 0.5) +  # Add horizontal lines at medians
  labs(x = "Pathology", y = "Area Percentage (%)", color = "Pathology") +
  scale_color_manual(values = c("#eb6a4d", "#6da9ed")) +  # Customize colors
  theme_minimal() +
  theme(
    panel.grid.major = element_line(linetype = "dashed"),  # Set major grid lines to dash line
    panel.grid.minor = element_line(linetype = "dashed"),   # Set minor grid lines to dash line
    axis.text.x = element_text(angle = 0, hjust = 0.5),  # Rotate x-axis labels for better readability
    plot.title = element_text(hjust = 0.5) # Set axis line to dash line
  ) +
  # Create the title with reversed order of "Benign" and "Malignant"
  ggtitle(paste("Cropped Area Per 598 * 598 for",formatted_total_dots_ROI," CBIS-DDSM Mass Images\nBenign: ", benign_count,", Malignant: ", malignant_count))


# Save the plot as a PDF with specified dimensions and resolution
ggsave("Cropped_area.pdf", ROI_area, width = 6.27, height = 5.7, dpi = 1000)
