//
//  SettingsComponents.swift
//  DynaMoE
//
//  Reusable presentation primitives for the Settings window, styled after the
//  macOS System Settings layout (grouped inset cards with a header banner).
//

import SwiftUI

/// Pane header shown at the top of a settings detail pane, following the macOS
/// System Settings pattern: a tinted icon tile, a bold pane title, and a short
/// descriptive subtitle, centered with no enclosing card.
struct SettingsSectionBanner: View {
    let icon: String
    let title: String
    let subtitle: String
    var tint: Color = .accentColor

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 30, weight: .medium))
                .foregroundStyle(.white)
                .frame(width: 58, height: 58)
                .background(tint.gradient)
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                .padding(.bottom, 4)

            Text(title)
                .font(.system(size: 20, weight: .bold))

            Text(subtitle)
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: 460)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 4)
        .padding(.bottom, 8)
    }
}

/// A grouped, rounded container holding a set of related settings rows, modeled
/// on System Settings' inset groups: a bold section label, a white card, and an
/// optional gray support line under the card.
struct SettingsCard<Content: View>: View {
    private let header: String?
    private let footer: String?
    private let spacing: CGFloat
    private let content: Content

    init(header: String? = nil, footer: String? = nil, spacing: CGFloat = 0, @ViewBuilder content: () -> Content) {
        self.header = header
        self.footer = footer
        self.spacing = spacing
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            if let header {
                Text(header)
                    .font(.system(size: 12.5, weight: .semibold))
                    .padding(.leading, 8)
            }

            VStack(alignment: .leading, spacing: spacing) {
                content
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(Color(nsColor: .controlBackgroundColor))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .stroke(Color.primary.opacity(0.08), lineWidth: 1)
            )

            if let footer {
                Text(footer)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.leading, 8)
            }
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

/// A standard settings row: an optional leading icon tile, a title with an
/// optional gray subtitle, and a trailing accessory (control, button, or value
/// label). Rows sit edge-to-edge inside a `SettingsCard`.
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
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 26, height: 26)
                    .background(iconTint.gradient)
                    .clipShape(RoundedRectangle(cornerRadius: 6.5, style: .continuous))
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 13))
                if let subtitle {
                    Text(subtitle)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Spacer(minLength: 12)

            trailing
        }
        .padding(.vertical, 8)
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

/// Pro-app numeric control (Final Cut/Motion inspector style): a slider for
/// feel paired with a narrow editable field for typing exact values. Values in
/// the field are clamped to the slider's range; there are no stepper arrows.
struct SettingsValueSlider: View {
    let value: Binding<Double>
    let range: ClosedRange<Double>
    let step: Double
    var fractionDigits: Int = 2
    var sliderWidth: CGFloat = 200
    var fieldWidth: CGFloat = 50

    var body: some View {
        HStack(spacing: 10) {
            Slider(value: clamped, in: range, step: step)
                .frame(width: sliderWidth)

            TextField("", value: clamped, format: .number.precision(.fractionLength(fractionDigits)))
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12, design: .monospaced))
                .multilineTextAlignment(.trailing)
                .frame(width: fieldWidth)
        }
    }

    private var clamped: Binding<Double> {
        Binding(
            get: { value.wrappedValue },
            set: { value.wrappedValue = min(max($0, range.lowerBound), range.upperBound) }
        )
    }
}
