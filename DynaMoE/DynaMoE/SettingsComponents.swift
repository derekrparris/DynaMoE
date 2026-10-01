//
//  SettingsComponents.swift
//  DynaMoE
//
//  Reusable presentation primitives for the Settings window, styled after the
//  macOS System Settings layout (grouped inset cards with a header banner).
//

import SwiftUI

/// Large header banner shown at the top of a settings detail pane: a tinted icon
/// tile, a bold title, and a short descriptive subtitle.
struct SettingsSectionBanner: View {
    let icon: String
    let title: String
    let subtitle: String
    var tint: Color = .accentColor

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 26, weight: .semibold))
                .foregroundColor(.white)
                .frame(width: 54, height: 54)
                .background(tint.gradient)
                .clipShape(RoundedRectangle(cornerRadius: 13, style: .continuous))

            Text(title)
                .font(.system(size: 20, weight: .bold))

            Text(subtitle)
                .font(.system(size: 12))
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: 520)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 22)
        .padding(.horizontal, 24)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor).opacity(0.6))
        )
    }
}

/// A grouped, rounded container holding a set of related settings rows.
struct SettingsCard<Content: View>: View {
    private let header: String?
    private let spacing: CGFloat
    private let content: Content

    init(header: String? = nil, spacing: CGFloat = 0, @ViewBuilder content: () -> Content) {
        self.header = header
        self.spacing = spacing
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let header {
                Text(header.uppercased())
                    .font(.system(size: 10.5, weight: .semibold))
                    .foregroundColor(.secondary)
                    .padding(.leading, 4)
            }

            VStack(alignment: .leading, spacing: spacing) {
                content
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color(nsColor: .controlBackgroundColor))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(Color.primary.opacity(0.06), lineWidth: 1)
            )
        }
    }
}

/// Inset separator used between rows inside a `SettingsCard`.
struct SettingsRowDivider: View {
    var body: some View {
        Divider()
            .opacity(0.5)
    }
}

/// A standard settings row: optional leading icon tile, a title with an optional
/// subtitle, and a trailing accessory (control, button, or value label).
struct SettingsRow<Trailing: View>: View {
    private let title: String
    private let subtitle: String?
    private let icon: String?
    private let iconTint: Color
    private let trailing: Trailing

    init(
        title: String,
        subtitle: String? = nil,
        icon: String? = nil,
        iconTint: Color = .accentColor,
        @ViewBuilder trailing: () -> Trailing
    ) {
        self.title = title
        self.subtitle = subtitle
        self.icon = icon
        self.iconTint = iconTint
        self.trailing = trailing()
    }

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            if let icon {
                Image(systemName: icon)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.white)
                    .frame(width: 26, height: 26)
                    .background(iconTint.gradient)
                    .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 13, weight: .medium))
                if let subtitle {
                    Text(subtitle)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Spacer(minLength: 12)

            trailing
        }
        .padding(.vertical, 10)
    }
}

/// Small monospaced value pill used for read-only trailing values.
struct SettingsValuePill: View {
    let text: String
    var tint: Color = .secondary

    var body: some View {
        Text(text)
            .font(.system(size: 11, weight: .medium, design: .monospaced))
            .foregroundColor(tint)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(tint.opacity(0.12))
            .cornerRadius(6)
    }
}

/// Compact capsule badge for state indicators (Default, Active, FlashMoE, ...).
/// Kept visually distinct from controls so status text never reads as a button.
struct SettingsStatusBadge: View {
    let text: String
    var systemImage: String? = nil
    var tint: Color = .secondary

    var body: some View {
        HStack(spacing: 3) {
            if let systemImage {
                Image(systemName: systemImage)
                    .font(.system(size: 8, weight: .bold))
            }
            Text(text.uppercased())
                .font(.system(size: 9.5, weight: .bold))
        }
        .foregroundColor(tint)
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(tint.opacity(0.12))
        .clipShape(Capsule())
        .fixedSize()
    }
}

/// Compact numeric input for dialing in values precisely: an editable value
/// field paired with a stepper. Replaces wide sliders for sampling settings.
struct SettingsNumericField: View {
    let value: Binding<Double>
    let range: ClosedRange<Double>
    let step: Double
    var fractionDigits: Int = 2
    var fieldWidth: CGFloat = 58

    var body: some View {
        HStack(spacing: 6) {
            TextField("", value: clamped, format: .number.precision(.fractionLength(fractionDigits)))
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12, design: .monospaced))
                .multilineTextAlignment(.trailing)
                .frame(width: fieldWidth)

            Stepper("", value: clamped, in: range, step: step)
                .labelsHidden()
        }
    }

    private var clamped: Binding<Double> {
        Binding(
            get: { value.wrappedValue },
            set: { value.wrappedValue = min(max($0, range.lowerBound), range.upperBound) }
        )
    }
}
